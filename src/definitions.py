class VM:
    def __init__(self, cpu, mem, price):
        self.cpu = cpu
        self.mem = mem
        self.cpu_now = cpu
        self.mem_now = mem
        self.price = price
        self.T = 0


class JOB:
    def __init__(self, arrival_time, j_id, j_type, cpu, mem, ex, duration):
        self.arrival_time = arrival_time
        self.id = j_id
        self.type = j_type
        self.cpu = cpu
        self.mem = mem
        self.ex = ex
        self.duration = duration
        self.ex_placementList = []
        self.running = False
        self.finished = False

