
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from functools import total_ordering

CENT = Decimal("0.01")


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────
class BankError(Exception):
    """Base class for all banking errors."""


class InsufficientFunds(BankError):
    pass


class WithdrawalLimitReached(BankError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Money: an immutable value object full of operator dunders
# ─────────────────────────────────────────────────────────────────────────────
@total_ordering  # supplies <=, >, >= from __eq__ and __lt__
class Money:
    __slots__ = ("_amount", "currency")

    RATES = {"USD": Decimal("1"), "EUR": Decimal("0.92"), "GBP": Decimal("0.79")}
    SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}

    def __init__(self, amount, currency: str = "USD"):
        currency = currency.upper()
        if currency not in self.RATES:
            raise ValueError(f"Unsupported currency: {currency}")
        self._amount = Decimal(str(amount)).quantize(CENT, rounding=ROUND_HALF_UP)
        self.currency = currency

    # ---- alternative constructors (classmethods) ----------------------------
    @classmethod
    def zero(cls, currency: str = "USD") -> Money:
        return cls(0, currency)

    @classmethod
    def from_string(cls, text: str) -> Money:
        """'125.50 EUR' -> Money('125.50', 'EUR'); currency defaults to USD."""
        amount, _, currency = text.strip().partition(" ")
        return cls(amount, currency or "USD")

    # ---- helpers -------------------------------------------------------------
    @property
    def amount(self) -> Decimal:
        return self._amount

    def to(self, currency: str) -> Money:
        currency = currency.upper()
        if currency == self.currency:
            return self
        usd = self._amount / self.RATES[self.currency]
        return Money(usd * self.RATES[currency], currency)

    def _key(self) -> Decimal:
        """Canonical value used for equality/hash so 1 USD == 0.92 EUR."""
        return (self._amount / self.RATES[self.currency]).quantize(Decimal("0.000001"))

    # ---- representation ------------------------------------------------------
    def __repr__(self) -> str:
        return f"Money('{self._amount}', '{self.currency}')"

    def __str__(self) -> str:
        sign = "-" if self._amount < 0 else ""
        return f"{sign}{self.SYMBOLS[self.currency]}{abs(self._amount):,.2f}"

    def __format__(self, spec: str) -> str:
        if spec == "code":
            return f"{self._amount:,.2f} {self.currency}"
        return format(str(self), spec)  # lets f"{m:>12}" work

    # ---- comparison / hashing ----------------------------------------------
    def __eq__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        return self._key() == other._key()

    def __lt__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        return self._key() < other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __bool__(self) -> bool:
        return self._amount != 0

    # ---- arithmetic ------------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self._amount + other.to(self.currency)._amount, self.currency)

    def __radd__(self, other):  # makes sum([Money, ...]) work (starts from 0)
        if isinstance(other, int) and other == 0:
            return self
        return NotImplemented

    def __sub__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self._amount - other.to(self.currency)._amount, self.currency)

    def __mul__(self, factor):
        if isinstance(factor, (int, float, Decimal)):
            return Money(self._amount * Decimal(str(factor)), self.currency)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float, Decimal)):
            return Money(self._amount / Decimal(str(divisor)), self.currency)
        return NotImplemented

    def __neg__(self) -> Money:
        return Money(-self._amount, self.currency)

    def __abs__(self) -> Money:
        return Money(abs(self._amount), self.currency)


# ─────────────────────────────────────────────────────────────────────────────
# Transaction record
# ─────────────────────────────────────────────────────────────────────────────
class Transaction:
    __slots__ = ("id", "kind", "amount", "balance_after")
    _ids = itertools.count(1)

    def __init__(self, kind: str, amount: Money, balance_after: Money):
        self.id = next(Transaction._ids)
        self.kind, self.amount, self.balance_after = kind, amount, balance_after

    def __repr__(self) -> str:
        return f"Transaction(#{self.id}, {self.kind!r}, {self.amount!r})"

    def __str__(self) -> str:
        return f"#{self.id:03d} {self.kind:<10} {self.amount:>10} -> balance {self.balance_after}"


# ─────────────────────────────────────────────────────────────────────────────
# Mixins (plain classes designed for cooperative multiple inheritance)
# ─────────────────────────────────────────────────────────────────────────────
class InterestMixin:
    """Adds monthly interest. Expects the host to be an Account."""

    annual_rate = Decimal("0.02")

    @staticmethod
    def monthly_rate(annual_rate: Decimal) -> Decimal:
        return annual_rate / 12

    @classmethod
    def project(cls, principal: Money, months: int) -> Money:
        """Compound projection; polymorphic because it reads cls.annual_rate."""
        return principal * ((1 + cls.monthly_rate(cls.annual_rate)) ** months)

    def apply_interest(self) -> None:
        interest = self.balance * self.monthly_rate(self.annual_rate)
        if interest:
            self._balance += interest
            self._record("interest", interest)

    def monthly_update(self) -> None:
        super().monthly_update()  # next class in the MRO
        self.apply_interest()


class AuditMixin:
    """Logs every deposit/withdrawal attempt by wrapping the next class in the MRO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # cooperative: forwards up the MRO
        self.audit_log: list[str] = []

    def _audit(self, message: str) -> None:
        self.audit_log.append(f"{len(self.audit_log) + 1:02d} {message}")

    def deposit(self, amount):
        self._audit(f"deposit requested: {amount}")
        result = super().deposit(amount)
        self._audit("deposit OK")
        return result

    def withdraw(self, amount):
        self._audit(f"withdraw requested: {amount}")
        try:
            result = super().withdraw(amount)
        except BankError as exc:
            self._audit(f"withdraw FAILED ({type(exc).__name__})")
            raise
        self._audit("withdraw OK")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Account hierarchy
# ─────────────────────────────────────────────────────────────────────────────
class Account(ABC):
    registry: dict[str, type[Account]] = {}
    account_type = "generic"
    max_withdrawals: int | None = None
    _id_counter = itertools.count(1001)

    # ---- class creation hook: every subclass registers itself ----------------
    def __init_subclass__(cls, /, account_type: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.account_type = account_type or cls.__name__.lower()
        Account.registry[cls.account_type] = cls

    def __init__(self, owner: str, balance: Money | None = None):
        self.owner = self.validate_owner(owner)
        self._balance = balance if balance is not None else Money.zero()
        self.id = next(Account._id_counter)
        self._history: list[Transaction] = []
        self._snapshots: list[tuple[Money, int, int]] = []
        self._withdrawals_this_month = 0

    # ---- classmethods: factories & introspection --------------------------------
    @classmethod
    def registered_types(cls) -> list[str]:
        return sorted(cls.registry)

    @classmethod
    def create(cls, kind: str, owner: str, *args, **kwargs) -> Account:
        try:
            klass = cls.registry[kind.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown account type {kind!r}. Known: {cls.registered_types()}"
            ) from None
        return klass(owner, *args, **kwargs)

    @classmethod
    def from_string(cls, text: str) -> Account:
        """'checking:alice smith:1500.00 USD' -> CheckingAccount(...)"""
        kind, owner, money = (part.strip() for part in text.split(":"))
        return cls.create(kind, owner, Money.from_string(money))

    # ---- staticmethods: pure utilities, no self/cls needed ---------------------------
    @staticmethod
    def validate_owner(name: str) -> str:
        cleaned = " ".join(name.split()).title()
        if len(cleaned) < 2 or not cleaned.replace(" ", "").isalpha():
            raise ValueError(f"Invalid owner name: {name!r}")
        return cleaned

    @staticmethod
    def is_business_day(day: date) -> bool:
        return day.weekday() < 5

    # ---- abstract interface ---------------------------------------------------------
    @abstractmethod
    def available_funds(self) -> Money:
        """Maximum amount that can be withdrawn right now."""

    @abstractmethod
    def monthly_fee(self) -> Money:
        """Fee charged at every month-end."""

    # ---- core behaviour -----------------------------------------------------------
    @property
    def balance(self) -> Money:
        return self._balance

    @property
    def currency(self) -> str:
        return self._balance.currency

    def _record(self, kind: str, amount: Money) -> None:
        self._history.append(Transaction(kind, amount, self._balance))

    def _normalize(self, amount) -> Money:
        money = amount if isinstance(amount, Money) else Money(amount, self.currency)
        money = money.to(self.currency)
        if money <= Money.zero(self.currency):
            raise ValueError("Amount must be positive")
        return money

    def deposit(self, amount) -> Account:
        money = self._normalize(amount)
        self._balance += money
        self._record("deposit", money)
        return self

    def withdraw(self, amount) -> Account:
        money = self._normalize(amount)
        if self.max_withdrawals is not None and self._withdrawals_this_month >= self.max_withdrawals:
            raise WithdrawalLimitReached(
                f"{self.owner}: limit of {self.max_withdrawals} withdrawals/month reached"
            )
        if money > self.available_funds():
            raise InsufficientFunds(
                f"{self.owner}: requested {money}, available {self.available_funds()}"
            )
        self._balance -= money
        self._withdrawals_this_month += 1
        self._record("withdrawal", money)
        return self

    def monthly_update(self) -> None:
        """Template method; mixins extend it cooperatively via super()."""
        self._withdrawals_this_month = 0
        fee = self.monthly_fee().to(self.currency)
        if fee:
            self._balance -= fee
            self._record("fee", fee)

    # ---- dunder methods ------------------------------------------------------------
    def __repr__(self) -> str:
        return f"{type(self).__name__}(owner={self.owner!r}, balance={self._balance!r}, id={self.id})"

    def __str__(self) -> str:
        return f"[{self.account_type:<9}] #{self.id} {self.owner:<12} {self._balance:>12}"

    def __eq__(self, other):  # identity of an account = its id
        if not isinstance(other, Account):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __lt__(self, other):  # ordering = by wealth (max()/sorted() need only this)
        if not isinstance(other, Account):
            return NotImplemented
        return self._balance < other._balance

    def __bool__(self) -> bool:  # truthy when the account holds positive funds
        return self._balance > Money.zero(self.currency)

    def __len__(self) -> int:  # number of transactions
        return len(self._history)

    def __iter__(self):
        return iter(self._history)

    def __getitem__(self, index):  # supports ints and slices
        return self._history[index]

    def __contains__(self, kind: str) -> bool:  # "fee" in account
        return any(tx.kind == kind for tx in self._history)

    def __add__(self, other):  # account + account -> combined Money
        if isinstance(other, Account):
            return self._balance + other._balance
        if isinstance(other, Money):
            return self._balance + other
        return NotImplemented

    def __radd__(self, other):  # lets sum(accounts, Money.zero()) work
        if isinstance(other, int) and other == 0:
            return self._balance
        if isinstance(other, Money):
            return other + self._balance
        return NotImplemented

    def __call__(self, amount):  # acct(Money(50)) deposits, acct(-Money(50)) withdraws
        money = amount if isinstance(amount, Money) else Money(amount, self.currency)
        if money > Money.zero(money.currency):
            return self.deposit(money)
        return self.withdraw(-money)

    def __enter__(self) -> Account:  # atomic block with rollback
        self._snapshots.append((self._balance, len(self._history), self._withdrawals_this_month))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        balance, history_len, withdrawals = self._snapshots.pop()
        if exc_type is not None:
            self._balance, self._withdrawals_this_month = balance, withdrawals
            del self._history[history_len:]
        return False  # never swallow the exception

    def __getattr__(self, name: str):  # dynamic: acct.balance_in_eur
        if name.startswith("balance_in_"):
            return self._balance.to(name.removeprefix("balance_in_"))
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


class CheckingAccount(Account, account_type="checking"):
    overdraft_limit = Money(500)

    def available_funds(self) -> Money:
        return self._balance + self.overdraft_limit.to(self.currency)

    def monthly_fee(self) -> Money:
        return Money(5)


class SavingsAccount(InterestMixin, Account, account_type="savings"):
    max_withdrawals = 3

    def available_funds(self) -> Money:
        return max(self._balance, Money.zero(self.currency))

    def monthly_fee(self) -> Money:
        return Money.zero()


class PremiumAccount(AuditMixin, SavingsAccount, account_type="premium"):
    annual_rate = Decimal("0.05")  # overrides InterestMixin.annual_rate
    max_withdrawals = 6
    tier = "gold"
    WELCOME_BONUS = Money(25)

    @classmethod
    def with_bonus(cls, owner: str, opening: Money) -> PremiumAccount:
        """Alternative constructor: opens the account and credits the bonus."""
        account = cls(owner, opening)
        account.deposit(cls.WELCOME_BONUS)
        return account

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f", tier={self.tier!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Bank: singleton container with container dunders
# ─────────────────────────────────────────────────────────────────────────────
class Bank:
    _instance: Bank | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, name: str = "PyBank"):
        if self._initialized:  # __init__ runs on every Bank(...) call
            return
        self.name = name
        self._accounts: dict[int, Account] = {}
        self._initialized = True

    # ---- classmethods / staticmethods --------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    @classmethod
    def open_from_manifest(cls, name: str, lines: list[str]) -> Bank:
        bank = cls(name)
        for line in lines:
            bank += Account.from_string(line)  # uses __iadd__
        return bank

    @staticmethod
    def next_business_day(day: date) -> date:
        while not Account.is_business_day(day):
            day += timedelta(days=1)
        return day

    @staticmethod
    def header(title: str) -> str:
        return f"\n{'=' * 64}\n{title}\n{'=' * 64}"

    # ---- behaviour ---------------------------------------------------------------------
    @property
    def total_assets(self) -> Money:
        return sum(self, Money.zero())  # iterates accounts via __iter__ + __radd__

    def run_month_end(self, on: date) -> date:
        effective = self.next_business_day(on)
        for account in self:
            account.monthly_update()
        return effective

    # ---- dunders ------------------------------------------------------------------------
    def __iadd__(self, account: Account) -> Bank:
        self._accounts[account.id] = account
        return self

    def __len__(self) -> int:
        return len(self._accounts)

    def __iter__(self):
        return iter(sorted(self._accounts.values(), key=lambda a: a.id))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._accounts[key]
        if isinstance(key, str):
            owner = Account.validate_owner(key)
            return tuple(a for a in self if a.owner == owner)
        raise TypeError("Use an account id (int) or an owner name (str)")

    def __contains__(self, item) -> bool:
        if isinstance(item, Account):
            return self._accounts.get(item.id) is item
        return item in self._accounts

    def __repr__(self) -> str:
        return f"Bank({self.name!r}, accounts={len(self)})"


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    Bank.reset()

    print(Bank.header("1. __init_subclass__ registry + abstract base class"))
    print("Registered types:", Account.registered_types())
    try:
        Account("Nobody")
    except TypeError as exc:
        print("Cannot instantiate the ABC ->", exc)

    print(Bank.header("2. Money: operator overloading"))
    a, b = Money("100"), Money("50", "EUR")
    print(f"a + b = {a + b} | a - b = {a - b} | a * 1.5 = {a * 1.5} | a / 4 = {a / 4}")
    print(f"-a = {-a} | abs(-a) = {abs(-a)} | bool(Money.zero()) = {bool(Money.zero())}")
    print("Money('1') == Money('0.92', 'EUR'):", Money("1") == Money("0.92", "EUR"))
    print("Set dedupes equal values:", len({Money("1"), Money("0.92", "EUR")}), "element")
    print(f"max(a, b) = {max(a, b):code} | sum([a, b, Money(10)]) = {sum([a, b, Money(10)])}")
    print("from_string:", repr(Money.from_string("19.99 gbp")))

    print(Bank.header("3. Multiple inheritance & MRO"))
    print(" -> ".join(cls.__name__ for cls in PremiumAccount.__mro__))

    print(Bank.header("4. Building the bank (classmethods + singleton)"))
    bank = Bank.open_from_manifest(
        "PyBank",
        [
            "checking:alice smith:1500.00 USD",
            "savings:bob jones:10000 USD",
            "premium:carol diaz:25000 EUR",
        ],
    )
    bank += PremiumAccount.with_bonus("dave lee", Money(1000))
    print(repr(bank), "| Bank() is bank:", Bank() is bank)
    for acct in bank:
        print(acct)
    print("repr of a premium account:", repr(bank[1004]))

    alice, bob, carol, dave = bank
    print(Bank.header("5. Callable objects, container dunders, in-place ops"))
    alice.deposit(200).withdraw(50)  # chained calls
    alice(Money(75))  # __call__ -> deposit
    alice(-Money(25))  # __call__ -> withdraw
    print(alice)
    print("len(alice) =", len(alice), "| 'fee' in alice:", "fee" in alice, "| 'deposit' in alice:", "deposit" in alice)
    print("alice[0]  :", alice[0])
    print("alice[-1] :", alice[-1])
    print("alice[1:3]:", alice[1:3])
    print("alice + bob =", alice + bob, "| bool(alice) =", bool(alice))

    print(Bank.header("6. Context manager: automatic rollback"))
    before = alice.balance
    try:
        with alice:
            alice.withdraw(1200)
            print("Inside block, balance:", alice.balance)
            alice.withdraw(5000)  # too much -> exception -> rollback
    except InsufficientFunds as exc:
        print("Error:", exc)
    print("After rollback:", alice.balance, "| unchanged:", alice.balance == before)

    print(Bank.header("7. Class-level limits on a subclass"))
    for i in range(1, 5):
        try:
            bob.withdraw(100)
            print(f"bob withdrawal {i}: OK")
        except WithdrawalLimitReached as exc:
            print(f"bob withdrawal {i}: blocked -> {exc}")

    print(Bank.header("8. AuditMixin (cooperative super) + __getattr__"))
    carol.deposit(500)
    try:
        carol.withdraw(10_000_000)
    except InsufficientFunds:
        pass
    carol.withdraw(100)
    print("\n".join(carol.audit_log))
    print("carol.balance_in_usd ->", carol.balance_in_usd, "| carol.balance_in_gbp ->", carol.balance_in_gbp)

    print(Bank.header("9. Polymorphic classmethod & staticmethod"))
    principal = Money(1000)
    print("SavingsAccount.project(1000, 12):", SavingsAccount.project(principal, 12))
    print("PremiumAccount.project(1000, 12):", PremiumAccount.project(principal, 12))
    print("InterestMixin.monthly_rate(6%)  :", round(InterestMixin.monthly_rate(Decimal("0.06")), 4))
    print("Is Sat 2026-10-31 a business day?", Account.is_business_day(date(2026, 10, 31)))

    print(Bank.header("10. Month-end run (template method through the MRO)"))
    effective = bank.run_month_end(date(2026, 10, 31))
    print("Effective processing date:", effective.strftime("%A, %d %B %Y"))
    for acct in sorted(bank, reverse=True):  # uses Account.__lt__
        print(acct)
    print("Alice's fee recorded?     ", "fee" in alice)
    print("Carol's last transaction: ", carol[-1])
    print("Total assets (USD)       :", bank.total_assets)
    print("Richest account          :", max(bank).owner)
    print("alice in bank / 9999 in bank:", alice in bank, 9999 in bank)
    print("bank['dave lee'] ->", bank["dave lee"])


if __name__ == "__main__":
    main()