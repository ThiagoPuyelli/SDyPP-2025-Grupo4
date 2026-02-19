import { useQuery } from "@tanstack/react-query";
import { fetchState } from "../../api/state";
import { SectionCard } from "../SectionCard";

export function StateSection() {
    const {
        data,
        // isLoading,
        isFetching,
        refetch,
        error,
        dataUpdatedAt,
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
            title="Estado del coordinador"
            loading={isFetching}
            onReload={refetch}
            lastUpdatedAt={dataUpdatedAt}
        >
            {error && <p>Error al cargar estado</p>}
            {data && (
                <>
                    <p>Estado: <strong>{data.state}</strong></p>
                    <p>Descripcion: {data.description}</p>
                    <p>Prefijo objetivo: {data.target_prefix}</p>
                </>
            )}
        </SectionCard>
    )
}