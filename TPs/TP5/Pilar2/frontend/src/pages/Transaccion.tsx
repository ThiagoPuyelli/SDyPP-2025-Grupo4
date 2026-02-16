import React, { useState } from "react";
import { signMessage } from "../utils/crypto";
import { Modal } from "../components/Modal";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div style={styles.field}>
            <label style={styles.label}>{label}</label>
            {children}
        </div>
    );
}

export default function Transaccion() {
    const [source, setSource] = useState("");
    const [target, setTarget] = useState("");
    const [amount, setAmount] = useState("");
    const [privateKey, setPrivateKey] = useState("");
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const [publishing, setPublishing] = useState(false);
    const [success, setSuccess] = useState<string | null>(null);

    async function publishTransaction(tx: any) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        try {
            const response = await fetch("https://blockchain.34.23.224.114.nip.io/tasks", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(tx),
                signal: controller.signal,
            });
            if (!response.ok) {
                const b = await response.json()
                throw new Error(`HTTP ${response.status} - ${b.detail}`);
            }
        } finally {
            clearTimeout(timeoutId);
        }
    }

    async function handleGenerate() {
        try {
            setError(null);

            const amountStr = Number(amount).toFixed(1);
            const timestamp = new Date().toISOString();

            const message = `${source}|${target}|${amountStr}|${timestamp}`;

            const signature = await signMessage(privateKey, message);

            setResult({
                source,
                target,
                amount: amountStr,
                timestamp,
                sign: signature,
            });
        } catch (e) {
            if (e instanceof Error && e.message) {
                setError(`Error: ${e.message}`);
            } else {
                setError("Error generando transacción");
            }
        }
    }

    async function handlePublish() {
        if (!result) return;

        try {
            setError(null);
            setPublishing(true);

            await publishTransaction(result);

            setResult(null);
            setSuccess("La transacción fue publicada correctamente");
        } catch (e) {
            setResult(null);
            if (e instanceof Error && e.message) {
                setError(`Error publicando transacción: ${e.message}`);
            } else {
                setError("Error publicando transacción");
            }
        } finally {
            setPublishing(false);
        }
    }

    return (
        <div style={styles.container}>
            <h1 style={styles.title}>Generador de transacciones</h1>

            <Field label="Public Key remitente">
                <textarea
                    style={textareaBase}
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                />
            </Field>

            <Field label="Public Key destino">
                <textarea
                    style={textareaBase}
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                />
            </Field>

            <Field label="Cantidad">
                <input
                    style={inputBase}
                    type="number"
                    step="0.0001"
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                />
            </Field>

            <Field label="Private Key remitente">
                <textarea
                    style={textareaBase}
                    value={privateKey}
                    onChange={(e) => setPrivateKey(e.target.value)}
                />
            </Field>

            <button style={styles.button} onClick={handleGenerate}>
                Generar transacción
            </button>

            {error && (
                <Modal
                    title="Error al generar transacción"
                    message={error}
                    variant="error"
                    onClose={() => setError(null)}
                />
            )}

            {/* {result && (
                <div style={styles.result}>
                    <div style={styles.resultTitle}>Payload generado</div>
                    <pre style={styles.pre}>
                        {JSON.stringify(result, null, 2)}
                    </pre>
                </div>
            )} */}
            {result && (
                <Modal
                    title="Transacción generada"
                    variant="success"
                    message={
                        <pre style={{ margin: 0 }}>
                            {JSON.stringify(result, null, 2)}
                        </pre>
                    }
                    onClose={() => setResult(null)}
                    actions={
                        <>
                            <button
    style={{
        ...styles.primaryButton,
        opacity: publishing ? 0.7 : 1,
        cursor: publishing ? "not-allowed" : "pointer",
    }}
    disabled={publishing}
    onClick={handlePublish}
>
    {publishing ? "Publicando..." : "Publicar"}
</button>
                        </>
                    }
                />
            )}
            {success && (
    <Modal
        title="Éxito"
        variant="success"
        message={success}
        onClose={() => setSuccess(null)}
    />
)}
        </div>
        
    );
}

const styles: Record<string, React.CSSProperties> = {
    container: {
        width: "100%",
        marginTop: "4rem",
        marginBottom: "2rem",
    },
    title: {
        fontSize: 24,
        marginBottom: 24,
    },
    sections: {
        display: "flex",
        gap: 16,
        flexWrap: "wrap",
        alignItems: "flex-start",
    },
    field: {
        display: "flex",
        flexDirection: "column",
        gap: 6,
        marginBottom: 16,
    },
    label: {
        fontSize: 13,
        fontWeight: 500,
        color: "#374151",
    },
    button: {
        outline: "none",
        marginTop: "1rem",
        marginBottom: "2rem",
        padding: "8px 14px",
        fontSize: 14,
        fontWeight: 500,
        borderRadius: 6,
        border: "1px solid #d1d5db",
        backgroundColor: "#f9fafb",
        cursor: "pointer",
    },
    result: {
        marginTop: 24,
        padding: 12,
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        backgroundColor: "#f9fafb",
    },
    resultTitle: {
        fontSize: 13,
        fontWeight: 600,
        marginBottom: 8,
    },
    pre: {
        fontSize: 12,
        backgroundColor: "#ffffff",
        padding: 12,
        borderRadius: 6,
        overflowX: "auto",
    },
    primaryButton: {
        padding: "6px 12px",
        fontSize: 13,
        borderRadius: 6,
        border: "1px solid #2563eb",
        backgroundColor: "#2563eb",
        color: "#ffffff",
        cursor: "pointer",
        outline: "none",
    },
};

const inputBase: React.CSSProperties = {
    width: "100%",
    padding: "8px 10px",
    fontSize: 14,
    borderRadius: 6,
    border: "1px solid #d1d5db",
    outline: "none",
    fontFamily: "inherit",
};

const textareaBase: React.CSSProperties = {
    ...inputBase,
    minHeight: 90,
    resize: "vertical",
    fontFamily: "monospace",
};