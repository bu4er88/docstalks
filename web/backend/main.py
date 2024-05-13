import argparse
import uvicorn


def main(app, host, port, reload):
    uvicorn.run(app=app, host=host, port=port, reload=reload)\
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Fetch links from a webpage')
    parser.add_argument('--app', default="app.api:app", type=str, help='URL of the webpage')
    parser.add_argument('--host', default="127.0.0.1", type=str)
    parser.add_argument('--port', default=8000, type=int)
    parser.add_argument('--reload', default=True, type=bool)

    args = parser.parse_args()
    print(args)
    app = args.app
    host = args.host
    port = args.port
    reload = args.reload

    main(app, host, port, reload)