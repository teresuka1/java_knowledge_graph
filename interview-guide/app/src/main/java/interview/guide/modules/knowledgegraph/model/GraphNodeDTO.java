package interview.guide.modules.knowledgegraph.model;

public record GraphNodeDTO(
    String id,
    String name,
    String type,
    String domain,
    Integer mentionCount,
    String sourceFile
) {
}
