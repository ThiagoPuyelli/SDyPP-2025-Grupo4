import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ======================
# Paths
# ======================

KEYS_DIR = Path("./keys")        # dentro de la imagen
OUTPUT_DIR = Path("./output")    # directorio montado al host
OUTPUT_DIR.mkdir(exist_ok=True)

SOURCE_PRIVATE = KEYS_DIR / "private_key_1.pem"
SOURCE_PUBLIC = KEYS_DIR / "public_key_1.pem"
TARGET_PUBLIC = KEYS_DIR / "public_key_4.pem"

amount = 2
timestamp = "string"

# ======================
# Cargar claves
# ======================

with open(SOURCE_PRIVATE, "rb") as f:
    private_key = serialization.load_pem_private_key(f.read(), password=None)

with open(SOURCE_PUBLIC, "r") as f:
    source_pub_pem = f.read()

with open(TARGET_PUBLIC, "r") as f:
    target_pub_pem = f.read()

# ======================
# Preparar mensaje
# ======================

amount_float = float(amount)
message = f"{source_pub_pem}|{target_pub_pem}|{amount_float}|{timestamp}".encode()

# ======================
# Firmar
# ======================

signature = private_key.sign(
    message,
    padding.PKCS1v15(),
    hashes.SHA256()
)

signature_b64 = base64.b64encode(signature).decode()


# ======================
# Debug: guardar message
# ======================

debug_file = OUTPUT_DIR / "out.txt"
with open(debug_file, "w") as f:
    f.write("=== Message string (antes de encode) ===\n")
    f.write(f"{source_pub_pem}|{target_pub_pem}|{amount_float}|{timestamp}\n\n")
    f.write("=== Message bytes (encode) ===\n")
    f.write(str(message))

print(f"✔ Debug message written at {debug_file}")



# ======================
# Crear payload
# ======================

payload = {
    "source": source_pub_pem,
    "target": target_pub_pem,
    "amount": amount_float,
    "timestamp": timestamp,
    "sign": signature_b64
}

# ======================
# Escribir archivo
# ======================

output_file = OUTPUT_DIR / "tx.json"
with open(output_file, "w") as f:
    json.dump(payload, f, indent=2)

print(f"✔ Transaction generated at {output_file}")
