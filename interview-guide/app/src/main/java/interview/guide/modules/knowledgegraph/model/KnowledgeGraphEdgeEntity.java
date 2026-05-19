package interview.guide.modules.knowledgegraph.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.LocalDateTime;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(
    name = "knowledge_graph_edges",
    indexes = {
        @Index(name = "idx_kg_edge_source", columnList = "sourceName"),
        @Index(name = "idx_kg_edge_target", columnList = "targetName"),
        @Index(name = "idx_kg_edge_domain", columnList = "domain"),
        @Index(name = "idx_kg_edge_relation", columnList = "relation")
    }
)
public class KnowledgeGraphEdgeEntity {

  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long id;

  @Column(nullable = false, length = 200)
  private String sourceName;

  @Column(nullable = false, length = 200)
  private String targetName;

  @Column(nullable = false, length = 100)
  private String relation;

  @Column(length = 100)
  private String sourceType;

  @Column(length = 100)
  private String targetType;

  @Column(length = 1000)
  private String evidence;

  @Column(length = 100)
  private String patternName;

  @Column(length = 200)
  private String sectionTitle;

  @Column(length = 100)
  private String sourceFile;

  @Column(nullable = false, length = 100)
  private String domain;

  private Double confidence;

  @Column(length = 100)
  private String method;

  @Column(nullable = false)
  private LocalDateTime createdAt;

  @PrePersist
  void onCreate() {
    createdAt = LocalDateTime.now();
  }
}
