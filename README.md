# Documentation for DocsTalks app

## 1. Installation
    ### Under development.. 

## 2. Run the vector database
Currently DocsTalks used Qdrant vector database. 
For running the database you must have Docker installed on your machine. 

**Run the database with Docker using the following command.**

*(This command ensures saving data of your database locally in folder ./qdrant_storage.)*
```
docker run -d --rm -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
Database will be up and running in detach mode, it means it will work in the backgroudn mode (if you want it running in the attached mode when you see all logs in your terminal, just delete flag: **-d** from the command above.

*Some usefull docker commands:*
```
docker ps                           # show list of running containers
docker ps -a                        # show list of all containers
docker stop <container-ID>          # stop the container
docker run <container-ID or name>   # run the containers
docker rm <container-ID or name>    # delete the container
docker logs <container-ID>          # show logs of the container
docker images                       # show list of images
docker rmi <image-ID>
```
When the database will be set up and running you cn 

## 3. Add documents in the vector database
### 3.1 config.yaml
Config.yaml is used to manage the process of procesisng documents and uploading data into the vector databse.

**Run this command to add files to the vector database:**
```
python load_docs_in_db.py --source <path_to_documents>
```

## 4. Run API server
**To run the server just run this command in your terminal:**
```
python api/backend/main.py
```
It runs the backend server. You can open it with the link http://127.0.0.1:8000/docs

**Follow this template to use API:**
```
http://127.0.0.1:8000/rag?<your-question>
```
It returns output in JSON format:
```
{
    "answer": <answer to the quetsion>,
    "sources": <list of sources were used>"
}
```
