export async function fetchState() {
    const res = await fetch("http://localhost:8000/state")
    if (!res.ok) {
        throw new Error("Failed to fetch state");
    }
    return res.json()
}
