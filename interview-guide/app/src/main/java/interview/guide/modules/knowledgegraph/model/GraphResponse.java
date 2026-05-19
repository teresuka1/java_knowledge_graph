package interview.guide.modules.knowledgegraph.model;

import java.util.List;

public record GraphResponse(
    List<GraphNodeDTO> nodes,
    List<GraphEdgeDTO> edges
) {
}
