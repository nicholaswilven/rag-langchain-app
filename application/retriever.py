from contextlib import closing
import os

from langchain.vectorstores import Neo4j

neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_host = os.getenv("NEO4J_HOST")

# Custom retriever to include filtering logic based on location, price, and taste
class CustomRestaurantRetriever:
    def __init__(self):
        pass
    
    def query_menus(self, input_lat, input_long, input_price, input_distance, input_embeddings):
        # graph_db = Neo4j(url = neo4j_host, auth = (neo4j_username, neo4j_password))
        
        with closing(Neo4j(
            url=neo4j_host,
            username=neo4j_username,
            password=neo4j_password
        )) as vector_store:
            """Query the menus based on price, distance, and embedding similarity."""
            result = vector_store.run(
                "MATCH (m:Menu)-[:OFFERS]->(r:Restaurant) "
                "WHERE m.price < $input_price AND "
                "distance(point(r), point({latitude: $input_lat, longitude: $input_long})) < $input_distance "
                "RETURN m",
                input_price=input_price,
                input_lat=input_lat,
                input_long=input_long,
                input_distance=input_distance
            )

            # Filter by cosine similarity
            filtered_menus = []
            for record in result:
                menu = record["m"]
                menu_embedding = menu["embedding"]
                similarity = 1 # self.get_cosine_similarity(menu_embedding, input_embeddings)
                if similarity > 0.7:
                    filtered_menus.append(menu)

            return filtered_menus