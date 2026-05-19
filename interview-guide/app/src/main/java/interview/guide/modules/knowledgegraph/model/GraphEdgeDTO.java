package interview.guide.modules.knowledgegraph.model;

public record GraphEdgeDTO(
    String id,
    String source,
    String target,
    String relation,
    Double confidence,
    String evidence,
    String sourceFile,
    String sectionTitle
) {
}
