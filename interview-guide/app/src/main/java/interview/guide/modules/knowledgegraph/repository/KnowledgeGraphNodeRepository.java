package interview.guide.modules.knowledgegraph.repository;

import interview.guide.modules.knowledgegraph.model.KnowledgeGraphNodeEntity;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface KnowledgeGraphNodeRepository
    extends JpaRepository<KnowledgeGraphNodeEntity, Long> {

  Optional<KnowledgeGraphNodeEntity> findByDomainAndName(String domain, String name);

  List<KnowledgeGraphNodeEntity> findByDomainAndNameIn(String domain, Collection<String> names);

  @Query("""
      SELECT n FROM KnowledgeGraphNodeEntity n
      WHERE (:domain IS NULL OR n.domain = :domain)
      ORDER BY n.mentionCount DESC, n.name ASC
      """)
  List<KnowledgeGraphNodeEntity> findTopNodes(
      @Param("domain") String domain,
      Pageable pageable
  );

  @Query("""
      SELECT n FROM KnowledgeGraphNodeEntity n
      WHERE (:domain IS NULL OR n.domain = :domain)
        AND LOWER(n.name) LIKE LOWER(CONCAT('%', :keyword, '%'))
      ORDER BY n.mentionCount DESC, n.name ASC
      """)
  List<KnowledgeGraphNodeEntity> searchNodes(
      @Param("domain") String domain,
      @Param("keyword") String keyword,
      Pageable pageable
  );

  @Query("SELECT n FROM KnowledgeGraphNodeEntity n WHERE n.name IN :names")
  List<KnowledgeGraphNodeEntity> findByNameIn(@Param("names") Collection<String> names);

  @Query("SELECT DISTINCT n.domain FROM KnowledgeGraphNodeEntity n ORDER BY n.domain")
  List<String> findAllDomains();
}
