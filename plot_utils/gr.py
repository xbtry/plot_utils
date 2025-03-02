import numpy as np

class vec3:
    def __init__(self, x, y, z):
        if isinstance(x, (tuple, list)):
            self.elems = np.array(x[0], x[1], x[2])
        elif isinstance(x, np.ndarray) and x.shape == (3,):
            self.elems = x
        else:
            self.elems = np.array([x,y,z])
    def __repr__(self):
        return f"Vec3({self.elems[0]}, {self.elems[1]}, {self.elems[2]})"
    
    def __add__(self, other):
        return vec3(*(self.elems + other.elems))

    def __mul__(self, scalar):
        return vec3(*(self.elems * scalar))

    def mag(self):
        return np.linalg.norm(self.elems)
    
    def scale(self, scalar):
        return vec3(*self.elems * scalar)
    
    def norm(self):
        mag = self.mag()
        if mag != 0:
            return vec3(*(self.elems / mag))
        return vec3(0,0,0)

    def dot(self, other):
        return np.dot(self.elems, other.elems)

    def cross(self, other):
        return vec3(*np.cross(self.elems, other.elems))

    def angle(self, other):
        dot_product = self.dot(other)
        mag1 = self.mag()
        mag2 = other.mag()
        if mag1 == 0 or mag2 == 0:
            raise ValueError("Cannot compute angle with zero magnitude vector.")
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def rotateX(self, angle):
        rotation_matrix = np.array([
            [1,0,0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        new_elems = rotation_matrix @ self.elems
        new_elems[np.isclose(new_elems, 0, atol=1e-10)] = 0
        return vec3(*new_elems)

    def rotateY(self, angle):
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0,1,0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        new_elems = rotation_matrix @ self.elems
        new_elems[np.isclose(new_elems, 0, atol=1e-10)] = 0
        return vec3(*new_elems)

    def rotateZ(self, angle):
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0,0,1]
        ])
        new_elems = rotation_matrix @ self.elems
        new_elems[np.isclose(new_elems, 0, atol=1e-10)] = 0
        return vec3(*new_elems)


class vec2:
    def __init__(self, x, y):
        if isinstance(x, (tuple, list)):
            self.elems = np.array(x[0], x[1])
        elif isinstance(x, np.ndarray) and x.shape == (2,):
            self.elems = x
        else:
            self.elems = np.array([x,y])

    def __repr__(self):
        return f"Vec2({self.elems[0]}, {self.elems[1]})"

    def __add__(self, other):
        return vec2(*(self.elems + other.elems))
    
    def __mul__(self, scalar):
        return vec2(*(self.elems * scalar)) 

    def scale(self, scalar):
        return vec2(*self.elems * scalar)
    
    def mag(self):
        return np.linalg.norm(self.elems)
    
    def norm(self):
        mag = self.mag()
        if mag != 0:
            return vec2(*(self.elems/mag))
        return vec2(0,0)
    
    def dot(self, other):
        return np.dot(self.elems, other.elems)

    def cross(self, other):
        return self.elems[0] * other.elems[1] - self.elems[1] * other.elems[0]
    def angle(self, other):
        dot_prod = self.dot(other)
        mag1 = self.mag()
        mag2 = other.mag()
        cos_theta = dot_prod / (mag1 * mag2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)
    
    def rotate(self, angle):
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        new_elems = rotation_matrix @ self.elems
        new_elems[np.isclose(new_elems, 0, atol=1e-10)] = 0
        return vec2(*new_elems)
