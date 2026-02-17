export async function fetchState() {
    const res = await fetch("api/state")
    if (!res.ok) {
        throw new Error("Failed to fetch state");
    }
    return res.json()
}
