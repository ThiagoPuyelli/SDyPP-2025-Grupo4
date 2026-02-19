import { useState, useEffect } from "react";
import type { ReactNode, CSSProperties } from "react";

type SectionCardProps = {
    title: string;
    loading?: boolean;
    onReload?: () => void;
    lastUpdatedAt?: number;
    children: ReactNode;
};

function formatTimeAgo(timestamp?: number) {
    if (!timestamp) return "";

    const seconds = Math.floor((Date.now() - timestamp) / 1000);

    if (seconds < 5) return "recién";
    if (seconds < 60) return `hace ${seconds}s`;

    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `hace ${minutes}m`;

    const hours = Math.floor(minutes / 60);
    return `hace ${hours}h`;
}

export function SectionCard({ title, loading, onReload, lastUpdatedAt, children }: SectionCardProps) {
    const [collapsed, setCollapsed] = useState(false);

    const [, forceUpdate] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            forceUpdate((n) => n + 1);
        }, 1000);

        return () => clearInterval(interval);
    }, []);

    return (
        <div style={styles.card}>
            <div style={styles.header}>
                <span style={styles.title}>{title}</span>

                <div style={styles.actions}>
                    <span style={styles.updated}>
                        {formatTimeAgo(lastUpdatedAt)}
                    </span>
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
        padding: "0.5rem 1rem 0.5rem 0",
    },
    actions: {
        display: "flex",
        alignItems: "center",
        gap: "1.5rem",
    },
    collapseButton: {
        background: "none",
        border: "none",
        cursor: "pointer",
        fontSize: 14,
        outline: "none",
        padding: "0.5rem 0",
    },
    body: {
        padding: 12,
    },
    updated: {
        fontSize: 12,
        color: "#6b7280",
    },
};
