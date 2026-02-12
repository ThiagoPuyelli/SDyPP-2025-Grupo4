function pemToArrayBuffer(pem: string): ArrayBuffer {
    const b64 = pem
        .replace(/-----BEGIN [^-]+-----/, "")
        .replace(/-----END [^-]+-----/, "")
        .replace(/\s/g, "");

    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }

    return bytes.buffer;
}

export async function signMessage(
    privateKeyPem: string,
    message: string
): Promise<string> {
    const keyBuffer = pemToArrayBuffer(privateKeyPem);

    const cryptoKey = await window.crypto.subtle.importKey(
        "pkcs8",
        keyBuffer,
        {
            name: "RSASSA-PKCS1-v1_5",
            hash: "SHA-256",
        },
        false,
        ["sign"]
    );

    const signature = await window.crypto.subtle.sign(
        "RSASSA-PKCS1-v1_5",
        cryptoKey,
        new TextEncoder().encode(message)
    );

    return btoa(
        String.fromCharCode(...new Uint8Array(signature))
    );
}
