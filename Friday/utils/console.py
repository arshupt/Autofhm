from rich.console import Console as xConsole
from rich.progress import Progress

class Console:
    def __init__(self):
        self.c = xConsole()
        self._pb = Progress(console=self.c)
        self._main_pb = None
        
        self._pb.start()

    def update_current(self,new_status):
        if self._main_pb :
            self._pb.update(self._main_pb, description=new_status)
        else:
            self.c.log("No Progress bar found")

    def log(self,st):
        self.c.log(st)

    def print(self, st):
        self.c.print(st)

    def start_pb(self, st):
        if self._main_pb is not None:
            self.c.log("Improper invokation!")
        self._main_pb = self._pb.add_task(st,start=False,total=100.0)

    def advance(self, to):
        self._pb.update(self._main_pb,advance=to)


    def stop_pb(self):
        self._pb.update(self._main_pb, completed=100.0)
        self._pb.stop_task(self._main_pb)
        self._main_pb = None

    def start_pb_loading(self):
        if self._main_pb:
            self._pb.start_task(self._main_pb)
            self._pb.update(self._main_pb)
        else:
            self.c.log("No Progress bar found")