import { useQuery } from "@tanstack/react-query";
import { fetchState } from "../../api/state";
import { SectionCard } from "../SectionCard";

export function StateSection() {
    const {
        data: state,
        // isLoading,
        isFetching,
        refetch,
        error,
    } = useQuery({
        queryKey: ["state"],
        queryFn: fetchState,
        select: (data: any) => ({
            state: data["state"],
            description: data["description"],
            target_prefix: data["target-prefix"],
        }),
    });

    return (

        <SectionCard
            title="Estado del nodo"
            loading={isFetching}
            onReload={refetch}
        >
            {error && <p>Error al cargar estado</p>}
            {state && (
                <>
                    <p>Estado: <strong>{state.state}</strong></p>
                    <p>Descripcion: {state.description}</p>
                    <p>Prefijo objetivo: {state.target_prefix}</p>
                </>
            )}
        </SectionCard>
    )
}