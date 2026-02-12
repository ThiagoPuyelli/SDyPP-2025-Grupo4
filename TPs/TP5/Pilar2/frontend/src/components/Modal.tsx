type ModalVariant = "default" | "success" | "error";

type ModalProps = {
    title: string;
    message: React.ReactNode;
    variant?: ModalVariant;
    onClose: () => void;
    actions?: React.ReactNode;
};


export function Modal({
    title,
    message,
    variant = "default",
    onClose,
    actions,
}: ModalProps) {
    return (
        <div style={styles.overlay} >
            <div
                style={{
                    ...styles.modal,
                }}
                onClick={(e) => e.stopPropagation()}
            >
                <div style={{
                    ...styles.header,
                    ...variantStyles[variant],
                }}>
                    <span>{title}</span>
                </div>

                <div style={styles.body}>
                    {message}
                </div>

                <div style={styles.footer}>
                    {actions ?? actions} 
                        <button style={styles.button} onClick={onClose}>
                            Cerrar
                        </button>

                </div>
            </div>
        </div>
    );
}


const styles: Record<string, React.CSSProperties> = {
    overlay: {
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(0, 0, 0, 0.35)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
    },
    modal: {
        width: 420,
        maxHeight: "80vh",
        backgroundColor: "#ffffff",
        borderRadius: 8,
        border: "1px solid #e5e7eb",
        boxShadow: "0 10px 25px rgba(0,0,0,0.15)",
        display: "flex",
        flexDirection: "column",
    },
    header: {
        padding: "8px 12px",
        fontSize: 14,
        fontWeight: 600,
        borderBottom: "1px solid #e5e7eb",
        backgroundColor: "#f9fafb",
    },
    body: {
        padding: 12,
        fontSize: 13,
        color: "#374151",
        overflow: "auto",
        whiteSpace: "pre",
        flex: 1,
    },
    footer: {
        padding: 8,
        display: "flex",
        justifyContent: "flex-end",
        gap: 8,
        borderTop: "1px solid #e5e7eb",
        backgroundColor: "#f9fafb",
    },
    button: {
        padding: "6px 12px",
        fontSize: 13,
        borderRadius: 6,
        border: "1px solid #d1d5db",
        backgroundColor: "#ffffff",
        cursor: "pointer",
        outline: "none",
    },
};

const variantStyles: Record<ModalVariant, React.CSSProperties> = {
    default: {},
    success: {
        // borderColor: "#86efac",
        backgroundColor: "#86efac",
    },
    error: {
        // borderColor: "#fca5a5",
        backgroundColor: "#fca5a5",
    },
};