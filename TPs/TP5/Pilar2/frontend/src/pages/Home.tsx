import { StateSection } from "../components/sections/StateSection";

export default function Home() {
    return (
        <div style={styles.container}>
            <h1 style={styles.title}>Blockchain Dashboard</h1>

            <div style={styles.sections}>
                <StateSection />
            </div>
        </div>
    );
}

const styles: Record<string, React.CSSProperties> = {
    container: {
        width: "100%",
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
};
