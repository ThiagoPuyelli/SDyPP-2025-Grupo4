import argparse
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


KEYS_DIR = Path("./keys")
BALANCES_FILE = Path("./output/balances.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Envía transacciones periódicas al coordinador para mantener activa la blockchain."
    )
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="Base URL del coordinador")
    parser.add_argument("--endpoint", default="/tasks", help="Endpoint de recepción de transacciones")
    parser.add_argument("--batch-size", type=int, default=4, help="Cantidad de transacciones por ciclo")
    parser.add_argument("--interval-seconds", type=float, default=15.0, help="Espera entre ciclos")
    parser.add_argument("--max-loops", type=int, default=0, help="0 = infinito")
    parser.add_argument("--balances-file", default=str(BALANCES_FILE), help="Archivo de saldos por key_id")
    parser.add_argument(
        "--amount-pattern",
        default="10,3,5,2,1",
        help="Montos rotativos separados por coma (ej: 10,3,5,2)",
    )
    parser.add_argument("--initial-balance-key1", type=float, default=10000.0)
    parser.add_argument("--min-amount", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--reset-balances", action="store_true", help="Ignora balances.json y reinicia saldo")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_amount_pattern(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("amount-pattern debe tener al menos un valor")
    return values


def discover_account_ids(keys_dir: Path) -> list[int]:
    account_ids = []
    for private_key_file in keys_dir.glob("private_key_*.pem"):
        suffix = private_key_file.stem.replace("private_key_", "")
        if suffix.isdigit():
            account_ids.append(int(suffix))
    return sorted(set(account_ids))


def load_private_key(path: Path):
    with path.open("rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def load_public_key(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_accounts(keys_dir: Path, account_ids: list[int]) -> dict[int, dict]:
    accounts: dict[int, dict] = {}
    for account_id in account_ids:
        private_path = keys_dir / f"private_key_{account_id}.pem"
        public_path = keys_dir / f"public_key_{account_id}.pem"
        if not private_path.exists() or not public_path.exists():
            continue

        accounts[account_id] = {
            "private_key": load_private_key(private_path),
            "public_pem": load_public_key(public_path),
        }
    return accounts


def sign_payload(source_private_key, source_public_pem: str, target_public_pem: str, amount: float, timestamp: str) -> str:
    message = f"{source_public_pem}|{target_public_pem}|{amount}|{timestamp}".encode()
    signature = source_private_key.sign(
        message,
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def build_payload(source_account: dict, target_account: dict, amount: float) -> dict:
    amount_float = float(amount)
    timestamp = datetime.now(timezone.utc).isoformat()
    source_public_pem = source_account["public_pem"]
    target_public_pem = target_account["public_pem"]
    signature_b64 = sign_payload(
        source_private_key=source_account["private_key"],
        source_public_pem=source_public_pem,
        target_public_pem=target_public_pem,
        amount=amount_float,
        timestamp=timestamp,
    )
    return {
        "source": source_public_pem,
        "target": target_public_pem,
        "amount": amount_float,
        "timestamp": timestamp,
        "sign": signature_b64,
    }


def load_runtime_state(
    balances_file: Path,
    account_ids: list[int],
    initial_balance_key1: float,
    reset_balances: bool,
) -> tuple[dict[int, float], int, int, str]:
    balances = {account_id: 0.0 for account_id in account_ids}
    next_source = account_ids[0]
    total_sent = 0
    source = "new"

    if not reset_balances and balances_file.exists():
        try:
            payload = json.loads(balances_file.read_text(encoding="utf-8"))
            raw_balances = payload.get("balances", {})
            if isinstance(raw_balances, dict):
                for account_id in account_ids:
                    if str(account_id) in raw_balances:
                        balances[account_id] = float(raw_balances[str(account_id)])
            raw_next = payload.get("next_source_id")
            if isinstance(raw_next, int) and raw_next in account_ids:
                next_source = raw_next
            total_sent = int(payload.get("total_sent", 0))
            source = "file"
        except (ValueError, TypeError, json.JSONDecodeError):
            source = "invalid_file"

    if source in {"new", "invalid_file"} and 1 in balances:
        balances[1] = float(initial_balance_key1)

    return balances, next_source, total_sent, source


def save_runtime_state(
    balances_file: Path,
    balances: dict[int, float],
    next_source: int,
    total_sent: int,
) -> None:
    balances_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "balances": {str(key_id): round(value, 8) for key_id, value in sorted(balances.items())},
        "next_source_id": next_source,
        "total_sent": total_sent,
    }
    balances_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def send_transaction(url: str, payload: dict, timeout: float) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except error.URLError as exc:
        return 0, str(exc.reason)


def pick_source(preferred_source: int, account_ids: list[int], balances: dict[int, float], min_amount: float) -> int | None:
    if balances.get(preferred_source, 0.0) >= min_amount:
        return preferred_source

    for account_id in account_ids:
        if balances.get(account_id, 0.0) >= min_amount:
            return account_id
    return None


def pick_target(source_id: int, account_ids: list[int], offset: int) -> int:
    source_index = account_ids.index(source_id)
    target_index = (source_index + offset) % len(account_ids)
    if account_ids[target_index] == source_id:
        target_index = (target_index + 1) % len(account_ids)
    return account_ids[target_index]


def main() -> None:
    args = parse_args()
    balances_file = Path(args.balances_file)
    account_ids = discover_account_ids(KEYS_DIR)
    if not account_ids:
        raise RuntimeError(f"No se encontraron claves en {KEYS_DIR}")

    accounts = load_accounts(KEYS_DIR, account_ids)
    account_ids = sorted(accounts.keys())
    if len(account_ids) < 2:
        raise RuntimeError("Se necesitan al menos 2 cuentas para transferir")

    balances, next_source, total_sent, state_source = load_runtime_state(
        balances_file=balances_file,
        account_ids=account_ids,
        initial_balance_key1=args.initial_balance_key1,
        reset_balances=args.reset_balances,
    )

    amounts = parse_amount_pattern(args.amount_pattern)
    route_offsets = [1, 2, 1, 3]  # 1->2, 2->4, luego sigue variando
    loop = 0
    url = f"{args.host}{args.endpoint}"

    save_runtime_state(balances_file, balances, next_source, total_sent)

    print(
        f"Emisor iniciado: {url} | accounts={account_ids} | "
        f"batch={args.batch_size} | interval={args.interval_seconds}s | "
        f"balances_source={state_source} | balances_file={balances_file}"
    )

    while True:
        loop += 1
        print(f"\n=== Loop {loop} ===")
        sent_this_loop = 0

        for i in range(args.batch_size):
            source_id = pick_source(next_source, account_ids, balances, args.min_amount)
            if source_id is None:
                print("[warn] No hay cuentas con saldo suficiente para seguir")
                break

            offset = route_offsets[total_sent % len(route_offsets)]
            target_id = pick_target(source_id, account_ids, offset)

            desired_amount = amounts[total_sent % len(amounts)]
            amount = min(desired_amount, balances[source_id])
            if amount < args.min_amount:
                print(
                    f"[warn] key_{source_id} sin saldo mínimo: "
                    f"saldo={balances[source_id]:.4f} min={args.min_amount:.4f}"
                )
                next_source = target_id
                continue

            payload = build_payload(accounts[source_id], accounts[target_id], amount)

            label = f"tx#{total_sent + 1} batch#{i + 1} {source_id}->{target_id} amount={amount}"
            if args.dry_run:
                print(f"[dry-run] {label}")
                status_code = 200
                body = '{"status":"dry-run"}'
            else:
                status_code, body = send_transaction(url=url, payload=payload, timeout=args.timeout)

            if status_code == 200:
                balances[source_id] -= amount
                balances[target_id] += amount
                sent_this_loop += 1
                total_sent += 1
                next_source = target_id
                save_runtime_state(balances_file, balances, next_source, total_sent)
                print(f"[ok] {label}")
            else:
                print(f"[reject] {label} status={status_code} detail={body}")

        print(
            f"[summary] loop={loop} enviados={sent_this_loop} total={total_sent} "
            f"proximo_origen=key_{next_source}"
        )

        if args.max_loops > 0 and loop >= args.max_loops:
            print("Fin: se alcanzó max-loops")
            break

        save_runtime_state(balances_file, balances, next_source, total_sent)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
