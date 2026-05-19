package interview.guide.modules.knowledgegraph.model;

import java.util.List;

public record GraphNodeDetailDTO(
    GraphNodeDTO node,
    List<GraphEdgeDTO> relations
) {
}
