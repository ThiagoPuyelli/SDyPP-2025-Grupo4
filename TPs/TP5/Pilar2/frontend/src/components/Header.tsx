import type { CSSProperties } from "react";
import { NavLink } from "react-router-dom";

function NavItem({ to, label }: { to: string; label: string }) {
    return (
        <NavLink
            to={to}
            style={({ isActive }) => ({
                ...styles.navItem,
                ...(isActive ? styles.navItemActive : {}),
            })}
        >
            {label}
        </NavLink>
    );
}

export function Header() {
    const title = "Blockchain"

    return (
        <header style={styles.header}>
            <div style={styles.container}>
                <span style={styles.title}>{title}</span>
                <nav style={styles.nav}>
                    <NavItem to="/" label="Dashboard" />
                    <NavItem to="/transaccion" label="Transacción" />
                </nav>
            </div>
        </header>
    );
}

const styles: Record<string, CSSProperties> = {
    header: {
        width: "100%",
        backgroundColor: "#f9fafb",
        borderBottom: "1px solid #e5e7eb",
    },
    container: {
        maxWidth: 1200,
        margin: "0 auto",
        padding: "10px 16px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
    },
    title: {
        fontSize: 15,
        fontWeight: 600,
        color: "#111827",
    },
    nav: {
        display: "flex",
        gap: 12,
    },
    navItem: {
        fontSize: 14,
        textDecoration: "none",
        color: "#374151",
        padding: "4px 8px",
        borderRadius: 6,
    },
    navItemActive: {
        backgroundColor: "#e5e7eb",
        color: "#111827",
        fontWeight: 500,
    },
};