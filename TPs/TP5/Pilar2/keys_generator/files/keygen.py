from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import os

NUM_KEYS = 10
OUTPUT_DIR = "keys"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(1, NUM_KEYS + 1):
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    with open(f"{OUTPUT_DIR}/private_key_{i}.pem", "wb") as f:
        f.write(private_pem)

    public_key = private_key.public_key()

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    with open(f"{OUTPUT_DIR}/public_key_{i}.pem", "wb") as f:
        f.write(public_pem)

    print(f"Par {i} generado")

print("OK: todos los pares generados")
