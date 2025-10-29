import tkinter as tk
from tkinter import filedialog, messagebox
import math, heapq, itertools, sys

COLORS = {
    "bg": "white",
    "edge": "black",
    "node": "lightgray",
    "start": "#ff6b6b",
    "goal": "#4d7cff",
    "visited": "#37d67a",
    "frontier": "#ffd66b",
    "path": "#c77cff",
    "text": "black"
}

def load_graph_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()!=""]
    n = int(lines[0])
    coords = []
    idx = 1
    for i in range(n):
        x,y = lines[idx].split()
        coords.append((int(x),int(y)))
        idx += 1
    neighbors = {i:[] for i in range(n)}
    for i in range(n):
        parts = lines[idx].split()
        k = int(parts[0])
        for p in parts[1:1+k]:
            neighbors[i].append(int(p)-1)
        idx += 1
    edges = {}
    for i, nbrs in neighbors.items():
        for j in nbrs:
            a,b = min(i,j), max(i,j)
            if (a,b) not in edges:
                x1,y1 = coords[a]; x2,y2 = coords[b]
                edges[(a,b)] = math.hypot(x1-x2, y1-y2)
    return coords, edges

def reconstruct_path(came_from, start, goal):
    path = []
    cur = goal
    while True:
        path.append(cur)
        if cur == start:
            break
        cur = came_from.get(cur)
        if cur is None:
            return []
    return list(reversed(path))

def astar_generator(nodes, edges, start, goal, mode):
    counter = itertools.count()
    frontier = []
    frontier_set = set()
    came_from = {}
    gcost = {}
    closed = set()
    gcost[start] = 0.0
    h = math.hypot(nodes[start][0]-nodes[goal][0], nodes[start][1]-nodes[goal][1]) if goal is not None else 0.0
    f = h if mode=="best" else gcost[start]+h
    heapq.heappush(frontier, (f, next(counter), start, 0.0))
    frontier_set.add(start)
    yield set(), set(frontier_set), None
    while frontier:
        f,_,node,g = heapq.heappop(frontier)
        if node in closed:
            continue
        frontier_set.discard(node)
        closed.add(node)
        yield set(closed), set(frontier_set), node
        if node == goal:
            path = reconstruct_path(came_from, start, goal)
            yield set(closed), set(frontier_set), ("FOUND", path)
            return
        nbrs = []
        for (a,b),w in edges.items():
            if a==node: nbrs.append((b,w))
            elif b==node: nbrs.append((a,w))
        for nb,w in nbrs:
            tentative_g = gcost.get(node, float('inf')) + w
            if mode == "astar":
                if tentative_g < gcost.get(nb, float('inf')):
                    gcost[nb] = tentative_g
                    came_from[nb] = node
                    hnb = math.hypot(nodes[nb][0]-nodes[goal][0], nodes[nb][1]-nodes[goal][1]) if goal is not None else 0.0
                    fnb = hnb if mode=="best" else tentative_g + hnb
                    if nb not in frontier_set:
                        heapq.heappush(frontier, (fnb, next(counter), nb, tentative_g))
                        frontier_set.add(nb)
                    else:
                        for i,entry in enumerate(frontier):
                            if entry[2]==nb and fnb < entry[0]:
                                frontier[i]=(fnb,entry[1],entry[2],tentative_g)
                                heapq.heapify(frontier)
                                break
            else:
                if nb not in came_from and nb not in closed:
                    came_from[nb] = node
                    hnb = math.hypot(nodes[nb][0]-nodes[goal][0], nodes[nb][1]-nodes[goal][1]) if goal is not None else 0.0
                    fnb = hnb
                    heapq.heappush(frontier, (fnb, next(counter), nb, tentative_g))
                    frontier_set.add(nb)
        yield set(closed), set(frontier_set), None
    yield set(closed), set(), ("NOTFOUND", None)

class GraphApp:
    def __init__(self, root, nodes=None, edges=None):
        self.root = root
        self.root.title("Best-First / A* Graph Pathfinder")
        self.canvas_w = 900
        self.canvas_h = 700
        self.left = tk.Frame(root)
        self.left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(self.left, bg=COLORS["bg"], width=self.canvas_w, height=self.canvas_h)
        self.canvas.pack(fill="both", expand=True)
        self.right = tk.Frame(root, width=300)
        self.right.pack(side="right", fill="y")

        self.alg = tk.StringVar(value="astar")
        tk.Radiobutton(self.right, text="A* (g+h)", variable=self.alg, value="astar").pack(anchor="w", padx=6, pady=(6,0))
        tk.Radiobutton(self.right, text="Best-First (h)", variable=self.alg, value="best").pack(anchor="w", padx=6)

        tk.Label(self.right, text="Delay (ms):").pack(anchor="w", padx=6, pady=(10,0))
        self.delay = tk.IntVar(value=80)
        tk.Entry(self.right, textvariable=self.delay, width=8).pack(anchor="w", padx=6)

        btn_frame = tk.Frame(self.right)
        btn_frame.pack(fill="x", pady=8, padx=6)
        tk.Button(btn_frame, text="Start", command=self.start_search).pack(side="left", expand=True, fill="x")
        tk.Button(btn_frame, text="Reset", command=self.reset_search).pack(side="left", expand=True, fill="x")

        self.start_label = tk.Label(self.right, text="Start: -")
        self.start_label.pack(anchor="w", padx=6)
        self.goal_label = tk.Label(self.right, text="Goal: -")
        self.goal_label.pack(anchor="w", padx=6)

        self.status = tk.Label(self.right, text="", anchor="w")
        self.status.pack(fill="x", padx=6, pady=6)

        self.canvas.bind("<Button-1>", self.on_click)
        self.nodes = nodes or []
        self.edges = edges or {}
        self.node_items = {}
        self.edge_items = []
        self.node_r = 14
        self.start = None
        self.goal = None
        self.generator = None
        self.running = False
        self.frontier = set()
        self.closed = set()
        self.path = []
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        if self.nodes:
            self.prepare_transform()
            self.draw_graph()

    def prepare_transform(self):
        xs = [x for x,y in self.nodes]
        ys = [y for x,y in self.nodes]
        minx,maxx=min(xs),max(xs)
        miny,maxy=min(ys),max(ys)
        if maxx-minx==0: maxx=minx+1
        if maxy-miny==0: maxy=miny+1
        pad = 40
        self.scale = min((self.canvas_w-2*pad)/(maxx-minx), (self.canvas_h-2*pad)/(maxy-miny))
        self.offset_x = pad - minx*self.scale
        self.offset_y = pad - miny*self.scale

    def to_canvas(self, x,y):
        cx = x*self.scale + self.offset_x
        cy = self.canvas_h - (y*self.scale + self.offset_y)
        return cx,cy

    def draw_graph(self):
        self.canvas.delete("all")
        self.node_items.clear()
        self.edge_items.clear()
        for (a,b),w in self.edges.items():
            x1,y1 = self.to_canvas(*self.nodes[a])
            x2,y2 = self.to_canvas(*self.nodes[b])
            id = self.canvas.create_line(x1,y1,x2,y2, fill=COLORS["edge"], width=2)
            self.edge_items.append(((a,b), id))
        for i,(x,y) in enumerate(self.nodes):
            cx,cy = self.to_canvas(x,y)
            nid = self.canvas.create_oval(cx-self.node_r, cy-self.node_r, cx+self.node_r, cy+self.node_r, fill=COLORS["node"], outline="black", width=1)
            tid = self.canvas.create_text(cx,cy, text=str(i+1), fill=COLORS["text"])
            self.node_items[i] = (nid, tid)

    def on_click(self, event):
        x,y = event.x, event.y
        clicked = None
        for i,(nid,tid) in self.node_items.items():
            x1,y1,x2,y2 = self.canvas.coords(nid)
            if x1<=x<=x2 and y1<=y<=y2:
                clicked = i
                break
        if clicked is None:
            return
        if self.start is None:
            self.start = clicked
            self.start_label.config(text=f"Start: {clicked+1}")
        elif self.goal is None and clicked != self.start:
            self.goal = clicked
            self.goal_label.config(text=f"Goal: {clicked+1}")
        else:
            return
        self.update_node_colors()
        self.reset_search()

    def update_node_colors(self):
        for i,(nid,tid) in self.node_items.items():
            color = COLORS["node"]
            if i==self.start: color = COLORS["start"]
            if i==self.goal: color = COLORS["goal"]
            self.canvas.itemconfig(nid, fill=color)

    def start_search(self):
        if self.start is None or self.goal is None:
            messagebox.showinfo("Info","Select start and goal nodes first")
            return
        self.generator = astar_generator(self.nodes, self.edges, self.start, self.goal, "best" if self.alg.get()=="best" else "astar")
        self.running = True
        self.status.config(text=f"Running {self.alg.get().upper()}...")
        self.animate()

    def animate(self):
        if not self.running:
            return
        try:
            closed, frontier, current = next(self.generator)
        except StopIteration:
            self.generator = None
            self.running = False
            return
        self.closed = closed
        self.frontier = frontier
        if isinstance(current, tuple) and current and current[0] in ("FOUND","NOTFOUND"):
            if current[0]=="FOUND":
                self.path = current[1]
                for n in self.path:
                    nid,tid = self.node_items[n]
                    self.canvas.itemconfig(nid, fill=COLORS["path"])
                self.status.config(text=f"Path found ({len(self.path)} nodes)")
            else:
                self.status.config(text="No path found")
            self.generator = None
            self.running = False
            return
        self.draw_search_state(current)
        delay = max(10, int(self.delay.get()))
        self.root.after(delay, self.animate)

    def draw_search_state(self, current=None):
        for (a,b), id in self.edge_items:
            self.canvas.itemconfig(id, fill=COLORS["edge"], width=2)
        for i,(nid,tid) in self.node_items.items():
            color = COLORS["node"]
            if i in self.closed: color = COLORS["visited"]
            if i in self.frontier: color = COLORS["frontier"]
            if i==self.start: color = COLORS["start"]
            if i==self.goal: color = COLORS["goal"]
            if i in self.path: color = COLORS["path"]
            if current is not None and i==current: color = COLORS["frontier"]
            self.canvas.itemconfig(nid, fill=color)
        for i,(nid,tid) in self.node_items.items():
            self.canvas.tag_raise(tid)
        self.status.config(text=f"Expanded: {len(self.closed)} Frontier: {len(self.frontier)}")

    def reset_search(self):
        self.generator = None
        self.running = False
        self.frontier = set()
        self.closed = set()
        self.path = []
        self.status.config(text="")
        self.update_node_colors()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("graphfile", nargs="?")
    args = parser.parse_args()
    root = tk.Tk()
    nodes, edges = [], {}
    if args.graphfile:
        try:
            nodes, edges = load_graph_file(args.graphfile)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            sys.exit(1)
    else:
        root.withdraw()
        path = filedialog.askopenfilename(title="Select graph file", filetypes=[("Text files","*.txt"),("All files","*.*")])
        root.deiconify()
        if path:
            try:
                nodes, edges = load_graph_file(path)
            except Exception as e:
                messagebox.showerror("Error", str(e))
    app = GraphApp(root, nodes, edges)
    root.mainloop()

if __name__=="__main__":
    main()
