
import csv
import os
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(".env")

from neo4j import GraphDatabase

from application.embeddings import StellaEmbeddingModel
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_host = os.getenv("NEO4J_HOST")

def create_restaurant(tx, restaurant_id, name, lat, long):
    """Insert restaurant node into Neo4j."""
    query = (
        "MERGE (r:Restaurant {id: $id}) "
        "SET r.name = $name, r.lat = $lat, r.long = $long"
    )
    tx.run(query, id=restaurant_id, name=name, lat=lat, long=long)

def create_menu(tx, menu_id, name, price, description, embedding):
    """Insert menu node into Neo4j."""
    query = (
        "MERGE (m:Menu {id: $id}) "
        "SET m.name = $name, m.price = $price, m.description = $description, "
        "m.embedding = $embedding"
    )
    tx.run(query, id=menu_id, name=name, price=price, description=description, embedding=embedding)

def create_offer_relationship(tx, restaurant_id, menu_id):
    """Create relationship between restaurant and menu."""
    query = (
        "MATCH (r:Restaurant {id: $restaurant_id}), (m:Menu {id: $menu_id}) "
        "MERGE (r)-[:OFFERS]->(m)"
    )
    tx.run(query, restaurant_id=restaurant_id, menu_id=menu_id)

def load_data_to_neo4j():
    driver = GraphDatabase.driver(neo4j_host, auth=(neo4j_username, neo4j_password))

    # Open CSV files and insert data
    """ with open('./data/kg/restaurants.csv', mode='r', encoding='utf-8') as file:
        with driver.session() as session:
            reader = csv.DictReader(file)
            for row in tqdm(reader):
                restaurant_id = row['id']
                name = row['name']
                lat = float(row['lat'])
                long = float(row['lng'])
                session.write_transaction(create_restaurant, restaurant_id, name, lat, long)"""

    model = StellaEmbeddingModel()
    model.download_model()
    model.load_model()
    
    batch_size = 32
    with open('./data/kg/restaurant-menus.csv', mode='r', encoding='utf-8') as file:
        with driver.session() as session:
            batch = []
            desc_list = []
            menu_id = 0
            i = 0
            reader = csv.DictReader(file)
            for row in tqdm(reader):
                if i < batch_size:
                    restaurant_id = row['id']
                    name = row['name']
                    price = float(row['price'])
                    description = row['description']
                    batch.append((menu_id, name, price, description))
                    desc_list.append(name + " " + description)
                    session.write_transaction(create_offer_relationship, restaurant_id, menu_id)
                    menu_id += 1
                    i += 1
                else:
                    i = 0
                    batch = []
                    desc_list = []
                    embedding = model.encode(desc_list)
                    for idx, args in enumerate(batch):
                        session.write_transaction(create_menu, *args, embedding[idx])
            
    driver.close()
                
if __name__ == "__main__":
    load_data_to_neo4j()
    
