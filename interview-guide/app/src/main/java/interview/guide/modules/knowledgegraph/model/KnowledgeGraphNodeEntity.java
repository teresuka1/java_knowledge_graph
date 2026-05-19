package interview.guide.modules.knowledgegraph.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;
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
    name = "knowledge_graph_nodes",
    indexes = {
        @Index(name = "idx_kg_node_name", columnList = "name"),
        @Index(name = "idx_kg_node_domain", columnList = "domain"),
        @Index(name = "idx_kg_node_type", columnList = "type")
    },
    uniqueConstraints = {
        @UniqueConstraint(name = "uk_kg_node_domain_name", columnNames = {"domain", "name"})
    }
)
public class KnowledgeGraphNodeEntity {

  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long id;

  @Column(nullable = false, length = 200)
  private String name;

  @Column(length = 100)
  private String type;

  private Integer mentionCount;

  @Column(length = 1000)
  private String mentions;

  @Column(length = 100)
  private String sourceFile;

  @Column(nullable = false, length = 100)
  private String domain;

  @Column(nullable = false)
  private LocalDateTime createdAt;

  @Column(nullable = false)
  private LocalDateTime updatedAt;

  @PrePersist
  void onCreate() {
    LocalDateTime now = LocalDateTime.now();
    createdAt = now;
    updatedAt = now;
    if (mentionCount == null) {
      mentionCount = 0;
    }
  }

  @PreUpdate
  void onUpdate() {
    updatedAt = LocalDateTime.now();
  }
}
