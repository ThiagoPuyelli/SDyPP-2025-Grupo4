import React, { useMemo, useState } from "react";
import { signMessage } from "../utils/crypto";
import { Modal } from "../components/Modal";

type TransactionPayload = {
    source: string;
    target: string;
    amount: number;
    timestamp: string;
    sign: string;
};

type PublishProgress = {
    sent: number;
    total: number;
};

const TRANSACTION_URL = "https://blockchain.34.23.224.114.nip.io/tasks";
const REQUEST_TIMEOUT_MS = 5000;

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div style={styles.field}>
            <label style={styles.label}>{label}</label>
            {children}
        </div>
    );
}

function parsePositiveNumber(rawValue: string, fieldName: string): number {
    const normalized = rawValue.trim().replace(",", ".");
    const parsed = Number(normalized);

    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`${fieldName} debe ser un número positivo`);
    }

    return parsed;
}

function parseNonNegativeNumber(rawValue: string, fieldName: string): number {
    const normalized = rawValue.trim().replace(",", ".");
    const parsed = Number(normalized);

    if (!Number.isFinite(parsed) || parsed < 0) {
        throw new Error(`${fieldName} debe ser un número mayor o igual a 0`);
    }

    return parsed;
}

function parsePositiveInteger(rawValue: string, fieldName: string): number {
    const parsed = Number(rawValue);

    if (!Number.isInteger(parsed) || parsed <= 0) {
        throw new Error(`${fieldName} debe ser un entero positivo`);
    }

    return parsed;
}

function formatAmountForSignature(amount: number): string {
    const raw = amount.toString().toLowerCase();

    if (!raw.includes("e")) {
        return Number.isInteger(amount) ? `${raw}.0` : raw;
    }

    const [mantissa, exponent] = raw.split("e");
    const sign = exponent.startsWith("-") ? "-" : "+";
    const digits = exponent.replace(/^[-+]/, "").padStart(2, "0");

    return `${mantissa}e${sign}${digits}`;
}

function normalizePem(rawValue: string): string {
    const normalized = rawValue.replace(/\r\n/g, "\n").trim();
    if (!normalized) {
        return "";
    }
    // El backend compara keys por igualdad exacta de string.
    // Mantener newline final evita mismatch con claves persistidas (ej: génesis).
    return `${normalized}\n`;
}

function delay(ms: number): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

export default function Transaccion() {
    const [source, setSource] = useState("");
    const [target, setTarget] = useState("");
    const [amount, setAmount] = useState("");
    const [privateKey, setPrivateKey] = useState("");

    const [txCount, setTxCount] = useState("1");
    const [timestampStepSeconds, setTimestampStepSeconds] = useState("1");
    const [sendIntervalMs, setSendIntervalMs] = useState("0");
    const [startTimestamp, setStartTimestamp] = useState("");

    const [generatedTransactions, setGeneratedTransactions] = useState<TransactionPayload[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const [publishing, setPublishing] = useState(false);
    const [publishProgress, setPublishProgress] = useState<PublishProgress | null>(null);

    const generateButtonLabel = useMemo(() => {
        const count = Number(txCount);
        return Number.isInteger(count) && count > 1
            ? "Generar lote de transacciones"
            : "Generar transacción";
    }, [txCount]);

    const resultPreview = useMemo(() => {
        if (generatedTransactions.length === 0) {
            return null;
        }

        if (generatedTransactions.length === 1) {
            return generatedTransactions[0];
        }

        return {
            cantidad: generatedTransactions.length,
            incremento_timestamp_segundos: timestampStepSeconds,
            intervalo_envio_ms: sendIntervalMs,
            primer_payload: generatedTransactions[0],
            ultimo_payload: generatedTransactions[generatedTransactions.length - 1],
        };
    }, [generatedTransactions, sendIntervalMs, timestampStepSeconds]);

    async function publishTransaction(tx: TransactionPayload) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

        try {
            const response = await fetch(TRANSACTION_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(tx),
                signal: controller.signal,
            });

            if (!response.ok) {
                let detail = `HTTP ${response.status}`;

                try {
                    const body = (await response.json()) as { detail?: string };
                    if (body?.detail) {
                        detail = `HTTP ${response.status} - ${body.detail}`;
                    }
                } catch {
                    // Si no hay JSON de error, devolvemos solo el código HTTP.
                }

                throw new Error(detail);
            }
        } finally {
            clearTimeout(timeoutId);
        }
    }

    async function handleGenerate() {
        try {
            setError(null);
            setSuccess(null);

            const sourceNormalized = normalizePem(source);
            const targetNormalized = normalizePem(target);
            const privateKeyNormalized = normalizePem(privateKey);

            if (!sourceNormalized || !targetNormalized || !privateKeyNormalized) {
                throw new Error("Completá source, target y private key");
            }

            if (sourceNormalized === targetNormalized) {
                throw new Error("Source y target no pueden ser iguales");
            }

            const amountNumber = parsePositiveNumber(amount, "La cantidad por transacción");
            const amountForSignature = formatAmountForSignature(amountNumber);
            const quantity = parsePositiveInteger(txCount, "La cantidad de transacciones");
            const stepSeconds = parseNonNegativeNumber(
                timestampStepSeconds,
                "El incremento de timestamp"
            );

            const baseDate = startTimestamp ? new Date(startTimestamp) : new Date();
            if (Number.isNaN(baseDate.getTime())) {
                throw new Error("El timestamp inicial es inválido");
            }

            const transactions: TransactionPayload[] = [];
            for (let i = 0; i < quantity; i++) {
                const txDate = new Date(baseDate.getTime() + i * stepSeconds * 1000);
                const timestamp = txDate.toISOString();
                const message = `${sourceNormalized}|${targetNormalized}|${amountForSignature}|${timestamp}`;
                const signature = await signMessage(privateKeyNormalized, message);

                transactions.push({
                    source: sourceNormalized,
                    target: targetNormalized,
                    amount: amountNumber,
                    timestamp,
                    sign: signature,
                });
            }

            setGeneratedTransactions(transactions);
        } catch (e) {
            if (e instanceof Error && e.message) {
                setError(`Error generando transacción: ${e.message}`);
            } else {
                setError("Error generando transacción");
            }
        }
    }

    async function handlePublish() {
        if (generatedTransactions.length === 0) {
            return;
        }

        let sentCount = 0;

        try {
            const intervalMs = parseNonNegativeNumber(sendIntervalMs, "El intervalo de envío");

            setError(null);
            setSuccess(null);
            setPublishing(true);
            setPublishProgress({ sent: 0, total: generatedTransactions.length });

            const txs = [...generatedTransactions];

            for (const [index, tx] of txs.entries()) {
                await publishTransaction(tx);
                sentCount = index + 1;
                setPublishProgress({ sent: sentCount, total: txs.length });

                if (index < txs.length - 1 && intervalMs > 0) {
                    await delay(intervalMs);
                }
            }

            setGeneratedTransactions([]);
            setSuccess(
                txs.length === 1
                    ? "La transacción fue publicada correctamente"
                    : `Se publicaron ${txs.length} transacciones correctamente`
            );
        } catch (e) {
            if (e instanceof Error && e.message) {
                setError(
                    `Error publicando transacciones (${sentCount}/${generatedTransactions.length} enviadas): ${e.message}`
                );
            } else {
                setError(
                    `Error publicando transacciones (${sentCount}/${generatedTransactions.length} enviadas)`
                );
            }
        } finally {
            setPublishing(false);
            setPublishProgress(null);
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

            <Field label="Cantidad por transacción (ej: 0.02 o 0,02)">
                <input
                    style={inputBase}
                    type="text"
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                />
            </Field>

            <Field label="Cantidad de transacciones">
                <input
                    style={inputBase}
                    type="number"
                    min={1}
                    step={1}
                    value={txCount}
                    onChange={(e) => setTxCount(e.target.value)}
                />
            </Field>

            <Field label="Incremento de timestamp por tx (segundos)">
                <input
                    style={inputBase}
                    type="number"
                    min={0}
                    step={0.1}
                    value={timestampStepSeconds}
                    onChange={(e) => setTimestampStepSeconds(e.target.value)}
                />
            </Field>

            <Field label="Intervalo de envío entre tx (ms)">
                <input
                    style={inputBase}
                    type="number"
                    min={0}
                    step={100}
                    value={sendIntervalMs}
                    onChange={(e) => setSendIntervalMs(e.target.value)}
                />
            </Field>

            <Field label="Timestamp inicial (opcional)">
                <input
                    style={inputBase}
                    type="datetime-local"
                    value={startTimestamp}
                    onChange={(e) => setStartTimestamp(e.target.value)}
                />
            </Field>

            <Field label="Private Key remitente">
                <textarea
                    style={textareaBase}
                    value={privateKey}
                    onChange={(e) => setPrivateKey(e.target.value)}
                />
            </Field>

            <button style={styles.button} onClick={handleGenerate} disabled={publishing}>
                {generateButtonLabel}
            </button>

            {error && (
                <Modal
                    title="Error"
                    message={error}
                    variant="error"
                    onClose={() => setError(null)}
                />
            )}

            {generatedTransactions.length > 0 && resultPreview && (
                <Modal
                    title={
                        generatedTransactions.length === 1
                            ? "Transacción generada"
                            : "Lote de transacciones generado"
                    }
                    variant="success"
                    message={<pre style={{ margin: 0 }}>{JSON.stringify(resultPreview, null, 2)}</pre>}
                    onClose={() => setGeneratedTransactions([])}
                    actions={
                        <button
                            style={{
                                ...styles.primaryButton,
                                opacity: publishing ? 0.7 : 1,
                                cursor: publishing ? "not-allowed" : "pointer",
                            }}
                            disabled={publishing}
                            onClick={handlePublish}
                        >
                            {publishing
                                ? `Publicando ${publishProgress?.sent ?? 0}/${publishProgress?.total ?? generatedTransactions.length}...`
                                : generatedTransactions.length === 1
                                    ? "Publicar"
                                    : `Publicar lote (${generatedTransactions.length})`}
                        </button>
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
