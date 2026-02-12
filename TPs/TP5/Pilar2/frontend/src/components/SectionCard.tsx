import React from "react";

type SectionCardProps = {
    title: string;
    loading?: boolean;
    onReload?: () => void;
    children: React.ReactNode;
};

export function SectionCard({ title, loading, onReload, children }: SectionCardProps) {
    return (
        <div style={styles.card}>
            <div style={styles.header}>
                <span style={styles.title}>{title}</span>

                {onReload && (
                    <button style={styles.reloadButton} disabled={loading} onClick={onReload}>
                        {loading ? "…" : "↻"}
                    </button>
                )}
            </div>

            <div style={styles.body}>
                {loading && !children ? <span>Cargando...</span> : children}
            </div>
        </div>
    );
}

const styles: Record<string, React.CSSProperties> = {
    card: {
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        backgroundColor: "#ffffff",
        display: "inline-block",
        minWidth: 300,
        width: "100%"
    },
    header: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "6px 10px",
        borderBottom: "1px solid #e5e7eb",
        backgroundColor: "#f9fafb",
        borderTopLeftRadius: 8,
        borderTopRightRadius: 8,
    },
    title: {
        fontSize: 14,
        fontWeight: 600,
    },
    reloadButton: {
        background: "none",
        border: "none",
        cursor: "pointer",
        fontSize: 14,
        outline: "none",
    },
    body: {
        padding: 12,
    },
};
