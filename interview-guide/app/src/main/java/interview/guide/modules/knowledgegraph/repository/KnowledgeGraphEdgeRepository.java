package interview.guide.modules.knowledgegraph.repository;

import interview.guide.modules.knowledgegraph.model.KnowledgeGraphEdgeEntity;
import java.util.Collection;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface KnowledgeGraphEdgeRepository
    extends JpaRepository<KnowledgeGraphEdgeEntity, Long> {

  @Query("""
      SELECT e FROM KnowledgeGraphEdgeEntity e
      WHERE (:domain IS NULL OR e.domain = :domain)
        AND (e.sourceName IN :names OR e.targetName IN :names)
      ORDER BY e.confidence DESC
      """)
  List<KnowledgeGraphEdgeEntity> findEdgesTouching(
      @Param("domain") String domain,
      @Param("names") Collection<String> names,
      Pageable pageable
  );

  @Query("SELECT DISTINCT e.relation FROM KnowledgeGraphEdgeEntity e ORDER BY e.relation")
  List<String> findAllRelationTypes();
}
