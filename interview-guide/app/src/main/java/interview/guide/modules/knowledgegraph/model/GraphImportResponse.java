package interview.guide.modules.knowledgegraph.model;

public record GraphImportResponse(
    int nodeCount,
    int edgeCount,
    int skippedCount
) {
}
