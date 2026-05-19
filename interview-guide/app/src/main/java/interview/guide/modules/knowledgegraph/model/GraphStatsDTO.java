package interview.guide.modules.knowledgegraph.model;

import java.util.List;

public record GraphStatsDTO(
    long nodeCount,
    long edgeCount,
    long domainCount,
    List<String> domains,
    List<String> relationTypes
) {
}
