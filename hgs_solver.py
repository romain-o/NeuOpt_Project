import torch
import numpy as np
import multiprocessing
import pyvrp

def _solve_worker(args):
    """Worker function to solve a single CVRP instance using PyVRP.
    Args:
        depot is a 1D array of shape (2,) containing the (x, y) coordinates of the depot.   
        coords is a 2D array of shape (real_size, 2) containing the (x, y) coordinates of the customers.
        deliveries is a 1D array of shape (real_size,) containing the int demand of each customer.
        capacity is an int scalar representing the vehicle capacity.
        time_limit is the maximum time allowed for solving this instance (in seconds).
        
        deliveries, capacity and coordinates are already rescaled to the scale factor and converted to integers.
        
        Returns: the scaled cost of the solution found by PyVRP, or infinity if no feasible solution is found within the time limit.
        """
    depot, coords, demands, capacity, time_limit, scale_factor = args

    m = pyvrp.Model()
    
    real_size = coords.shape[0]

    depot = m.add_depot(
        x=depot[0], 
        y=depot[1], 
        name="Depot"
    )
    
    vehicles = m.add_vehicle_type(num_available=real_size, capacity=capacity)
    
    for i, demand in enumerate(demands):
        m.add_client(
            x=float(coords[i][0]),
            y=float(coords[i][1]),
            delivery=int(demand),
            name=f"Client {i + 1}",
        )

    for frm in m.locations:
        for to in m.locations:
            if frm != to:
                dist = int(np.linalg.norm(np.array([frm.x, frm.y]) - np.array([to.x, to.y])))
                m.add_edge(frm, to, distance=dist)

    res = m.solve(stop=pyvrp.stop.MaxRuntime(time_limit))
    
    cost = res.cost() if res.is_feasible() else float('inf')
    
    return cost / scale_factor


class HGSSolver:
    def __init__(self, problem, time_limit=10.0, scale_factor=1000):
        self.time_limit = time_limit
        self.scale_factor = scale_factor
        self.size = problem.size
        self.real_size = problem.real_size
        self.dummy_size = self.size - self.real_size

    def _tensor_to_numpy(self, t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.array(t)

    def __call__(self, batch, capacity=1.0):
        coords_ = batch['coordinates'] # shape: (batch_size, size, 2)
        demands_ = batch['demand'] # shape: (batch_size, size)
        
        depot = coords_[:, 0, :]
        
        coords = coords_[:, self.dummy_size:, :]
        demands = demands_[:, self.dummy_size:]

        depot = self._tensor_to_numpy(depot)*self.scale_factor
        depot = depot.astype(int)
        coords = self._tensor_to_numpy(coords)*self.scale_factor
        coords = coords.astype(int)
        demands = self._tensor_to_numpy(demands)*self.scale_factor
        demands = demands.astype(int)
        capacity = int(capacity*self.scale_factor)

        batch_size = depot.shape[0]
        tasks = []

        for i in range(batch_size):
            tasks.append((
                depot[i],
                coords[i],
                demands[i],
                capacity,
                self.time_limit,
                self.scale_factor
            ))

        from hgs_solver import _solve_worker
        
        results = [] # temporary way until multiprocessing is implemented
        for task in tasks:
            result = _solve_worker(task)
            results.append(result)
    
        return results