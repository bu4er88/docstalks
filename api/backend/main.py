import argparse
import uvicorn
from app.api import config


def main(app, host, port, reload):
    uvicorn.run(app=app, host=host, port=port, reload=reload)
    
    
if __name__=="__main__":
    server_host = str(config['server_host'])
    server_port = int(config['server_port'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--app',    default="app.api:app",  type=str)
    parser.add_argument('--host',   default=server_host,    type=str)
    parser.add_argument('--port',   default=server_port,    type=int)
    parser.add_argument('--reload', default=True,           type=bool)
    args = parser.parse_args()
    
    app = args.app
    host = args.host
    port = args.port
    reload = args.reload

    main(app, host, port, reload)