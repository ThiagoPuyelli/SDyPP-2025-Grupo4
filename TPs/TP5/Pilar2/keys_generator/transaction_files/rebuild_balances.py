import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


KEYS_DIR = Path("./keys")
DEFAULT_TX_STREAM_FILE = Path("./output/tx_stream.jsonl")
DEFAULT_BALANCES_FILE = Path("./output/balances.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruye saldos por key_id recorriendo tx_stream.jsonl."
    )
    parser.add_argument("--keys-dir", default=str(KEYS_DIR))
    parser.add_argument("--tx-stream-file", default=str(DEFAULT_TX_STREAM_FILE))
    parser.add_argument("--balances-file", default=str(DEFAULT_BALANCES_FILE))
    parser.add_argument("--initial-balance-key1", type=float, default=10000.0)
    parser.add_argument("--next-source-id", type=int, default=1)
    return parser.parse_args()


def discover_account_ids(keys_dir: Path) -> list[int]:
    account_ids = []
    for private_key_file in keys_dir.glob("private_key_*.pem"):
        suffix = private_key_file.stem.replace("private_key_", "")
        if suffix.isdigit():
            account_ids.append(int(suffix))
    return sorted(set(account_ids))


def load_public_key_map(keys_dir: Path, account_ids: list[int]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for account_id in account_ids:
        public_key_file = keys_dir / f"public_key_{account_id}.pem"
        if public_key_file.exists():
            mapping[public_key_file.read_text(encoding="utf-8")] = account_id
    return mapping


def rebuild_balances(
    tx_stream_file: Path,
    account_ids: list[int],
    public_key_to_id: dict[str, int],
    initial_balance_key1: float,
) -> tuple[dict[int, float], int, int, int]:
    balances = {account_id: 0.0 for account_id in account_ids}
    if 1 in balances:
        balances[1] = float(initial_balance_key1)

    total_lines = 0
    applied_transactions = 0
    ignored_transactions = 0

    if not tx_stream_file.exists():
        return balances, total_lines, applied_transactions, ignored_transactions

    with tx_stream_file.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                ignored_transactions += 1
                continue

            source_pem = payload.get("source")
            target_pem = payload.get("target")
            amount_raw = payload.get("amount")

            try:
                amount = float(amount_raw)
            except (TypeError, ValueError):
                ignored_transactions += 1
                continue

            touched = False
            source_id = public_key_to_id.get(source_pem)
            target_id = public_key_to_id.get(target_pem)

            if source_id is not None:
                balances[source_id] -= amount
                touched = True
            if target_id is not None:
                balances[target_id] += amount
                touched = True

            if touched:
                applied_transactions += 1
            else:
                ignored_transactions += 1

    return balances, total_lines, applied_transactions, ignored_transactions


def save_balances_file(
    balances_file: Path,
    balances: dict[int, float],
    tx_stream_file: Path,
    total_lines: int,
    applied_transactions: int,
    ignored_transactions: int,
    next_source_id: int,
) -> None:
    balances_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(tx_stream_file),
        "lines_read": total_lines,
        "applied_transactions": applied_transactions,
        "ignored_transactions": ignored_transactions,
        "balances": {str(key_id): round(value, 8) for key_id, value in sorted(balances.items())},
        "next_source_id": next_source_id,
        "total_sent": applied_transactions,
    }
    balances_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    keys_dir = Path(args.keys_dir)
    tx_stream_file = Path(args.tx_stream_file)
    balances_file = Path(args.balances_file)

    account_ids = discover_account_ids(keys_dir)
    if not account_ids:
        raise RuntimeError(f"No se encontraron claves en {keys_dir}")

    public_key_to_id = load_public_key_map(keys_dir, account_ids)
    balances, lines_read, applied, ignored = rebuild_balances(
        tx_stream_file=tx_stream_file,
        account_ids=account_ids,
        public_key_to_id=public_key_to_id,
        initial_balance_key1=args.initial_balance_key1,
    )

    next_source_id = args.next_source_id if args.next_source_id in account_ids else account_ids[0]

    save_balances_file(
        balances_file=balances_file,
        balances=balances,
        tx_stream_file=tx_stream_file,
        total_lines=lines_read,
        applied_transactions=applied,
        ignored_transactions=ignored,
        next_source_id=next_source_id,
    )

    print(f"Saldos reconstruidos en {balances_file}")
    print(f"Lineas leidas: {lines_read}")
    print(f"Transacciones aplicadas: {applied}")
    print(f"Transacciones ignoradas: {ignored}")
    print("Balances:")
    for key_id in sorted(balances):
        print(f"  key_{key_id}: {balances[key_id]:.8f}")


if __name__ == "__main__":
    main()
