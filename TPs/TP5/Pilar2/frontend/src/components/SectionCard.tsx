import { useState } from "react";
import type { ReactNode, CSSProperties } from "react";

type SectionCardProps = {
    title: string;
    loading?: boolean;
    onReload?: () => void;
    children: ReactNode;
};

export function SectionCard({ title, loading, onReload, children }: SectionCardProps) {
    const [collapsed, setCollapsed] = useState(false);

    return (
        <div style={styles.card}>
            <div style={styles.header}>
                <span style={styles.title}>{title}</span>

                <div style={styles.actions}>
                    <button
                        style={styles.collapseButton}
                        onClick={() => setCollapsed((c) => !c)}
                        aria-label={collapsed ? "Expandir" : "Colapsar"}
                    >
                        {collapsed ? "▸" : "▾"}
                    </button>

                {onReload && (
                    <button style={styles.reloadButton} disabled={loading} onClick={onReload}>
                        {loading ? "…" : "↻"}
                    </button>
                )}
                </div>
            </div>

            {!collapsed && (
                <div style={styles.body}>
                    {loading && !children ? <span>Cargando...</span> : children}
                </div>
            )}
        </div>
    );
}

const styles: Record<string, CSSProperties> = {
    card: {
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        backgroundColor: "#ffffff",
        display: "inline-block",
        minWidth: 300,
        width: "100%",
        maxHeight: "90vh",
        overflow: "auto",
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

        position: "sticky",
        top: 0,
        zIndex: 1,
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
    actions: {
        display: "flex",
        alignItems: "center",
        gap: 6,
    },
    collapseButton: {
        background: "none",
        border: "none",
        cursor: "pointer",
        fontSize: 14,
        outline: "none",
        padding: "0 2px",
    },
    body: {
        padding: 12,
    },
};
