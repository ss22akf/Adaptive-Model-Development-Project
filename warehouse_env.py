import numpy as np
from typing import Optional, Tuple, List


class WarehouseEnv:
    """
    Grid-based warehouse environment for RL.

    The agent must:
    - Navigate to pickup location
    - Deliver package to destination
    - Avoid obstacles
    - Manage battery effectively
    """

    # Action space (movement directions)
    ACTIONS = {
        0: (-1, 0),   # UP
        1: (1, 0),    # DOWN
        2: (0, -1),   # LEFT
        3: (0, 1),    # RIGHT
    }

    # Tunable parameters
    def __init__(
        self,
        size: int = 20,
        maxBattery: int = 60,
        maxSteps: int = 250,
        collisionLimit: int = 2,
        dynamicObstacles: bool = True,
        seed: Optional[int] = 42,
    ) -> None:

        # Environment configuration
        self.size = size
        self.maxBattery = maxBattery
        self.maxSteps = maxSteps
        self.collisionLimit = collisionLimit
        self.dynamicObstacles = dynamicObstacles

        # Random generator for reproducibility
        self.rng = np.random.default_rng(seed)

        # Fixed key positions
        self.startPos = (0, 0)
        self.pickupPos = (2, 2)
        self.deliveryPos = (18, 18)
        self.chargerPos = (10, 10)

        # Static obstacles (walls with openings)
        self.staticObstacles = set()
        for col in range(5, 15):
            if col != 9:
                self.staticObstacles.add((8, col))
            if col != 11:
                self.staticObstacles.add((14, col))

        # Dynamic obstacle path
        self.movingObstaclePath = [(5, 12), (5, 13), (5, 14), (5, 13)]
        self.movingIndex = 0

        self.actionSpace = len(self.ACTIONS)

        self.reset()


    # Reset Environment 
    def reset(self) -> Tuple[int, int, int, int]:
        """
        Reset environment state at the beginning of each episode.
        """
        self.robotPos = self.startPos
        self.battery = self.maxBattery
        self.hasPackage = 0
        self.steps = 0
        self.collisionCount = 0
        self.totalEnergyUsed = 0
        self.movingIndex = 0

        return self.getState()


    # State Representation
    def getState(self) -> Tuple[int, int, int, int]:
        """
        State = (x position, y position, package status, battery level bucket)
        """
        x, y = self.robotPos
        batteryRatio = self.battery / self.maxBattery

        # Discretise battery into 4 levels
        if batteryRatio <= 0.25:
            batteryBucket = 0
        elif batteryRatio <= 0.50:
            batteryBucket = 1
        elif batteryRatio <= 0.75:
            batteryBucket = 2
        else:
            batteryBucket = 3

        return (x, y, self.hasPackage, batteryBucket)

    # Obstacle handling
    def getObstacles(self) -> set:
        """
        Combine static and dynamic obstacles.
        """
        obstacles = set(self.staticObstacles)

        if self.dynamicObstacles:
            obstacles.add(self.movingObstaclePath[self.movingIndex])

        return obstacles

    def _advanceDynamicObstacle(self) -> None:
        """
        Move dynamic obstacle along predefined path.
        """
        if self.dynamicObstacles:
            self.movingIndex = (self.movingIndex + 1) % len(self.movingObstaclePath)

    # Helper functions
    def _manhattanDistance(self, a, b):
        """
        Compute Manhattan distance between two grid positions.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _currentTarget(self):
        """
        Determine current goal:
        - pickup (if package not collected)
        - delivery (if package collected)
        """
        return self.pickupPos if self.hasPackage == 0 else self.deliveryPos

    # Step function - RL Logic
    def step(self, action: int):
        """
        Execute one action and update environment state.
        """

        # Ensure valid action
        if action not in self.ACTIONS:
            raise ValueError("Invalid action")

        # Add stochasticity (action noise)
        if self.rng.random() < 0.08:
            action = int(self.rng.integers(0, self.actionSpace))

        dx, dy = self.ACTIONS[action]
        x, y = self.robotPos
        nx, ny = x + dx, y + dy

        reward = -1.0  # step penalty
        done = False

        # Update battery and steps
        self.battery -= 1
        self.steps += 1
        self.totalEnergyUsed += 1

        # Distance based shaping
        oldDist = self._manhattanDistance(self.robotPos, self._currentTarget())

        # Collision check
        if (
            nx < 0 or ny < 0 or
            nx >= self.size or ny >= self.size or
            (nx, ny) in self.getObstacles()
        ):
            self.collisionCount += 1
            reward -= 12.0
            newPos = self.robotPos
        else:
            self.robotPos = (nx, ny)
            newPos = self.robotPos

        # Reward shaping based on progress
        newDist = self._manhattanDistance(newPos, self._currentTarget())
        if newDist < oldDist:
            reward += 1.0
        elif newDist > oldDist:
            reward -= 0.5

        # Penalise proximity to obstacles (safety awareness)
        for dx, dy in self.ACTIONS.values():
            adj = (self.robotPos[0] + dx, self.robotPos[1] + dy)
            if adj in self.staticObstacles:
                reward -= 0.8

        # Pickup reward
        if self.robotPos == self.pickupPos and self.hasPackage == 0:
            self.hasPackage = 1
            reward += 25.0

        # Delivery reward
        if self.robotPos == self.deliveryPos and self.hasPackage == 1:
            reward += 100.0
            done = True

        # Charging behaviour
        if self.robotPos == self.chargerPos and self.battery < self.maxBattery:
            chargeGain = self.maxBattery - self.battery
            self.battery = self.maxBattery
            if chargeGain >= int(0.3 * self.maxBattery):
                reward += 6.0

        # Low battery penalty
        if self.battery <= int(0.2 * self.maxBattery) and self.robotPos != self.chargerPos:
            reward -= 4.0

        # Terminal conditions
        if self.battery <= 0:
            reward -= 60.0
            done = True

        if self.collisionCount >= self.collisionLimit:
            done = True

        if self.steps >= self.maxSteps:
            done = True

        # Update dynamic obstacle
        self._advanceDynamicObstacle()

        return self.getState(), reward, done

    # Path Visualisation helper function
    def getGreedyPath(self, qTable, maxSteps=None):
        """
        Generate path using greedy policy (for visualisation).
        """
        state = self.reset()
        path = [self.robotPos]
        done = False

        limit = maxSteps if maxSteps else self.maxSteps

        while not done and len(path) < limit:
            action = int(np.argmax(qTable[state]))
            state, _, done = self.step(action)
            path.append(self.robotPos)

        return path