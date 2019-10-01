from collections import deque


class Memory:
	def __init__(self, memory_size):
		self.memory_size = memory_size
		self.ltmemory = deque(maxlen=memory_size)
		self.stmemory = deque(maxlen=memory_size)

	def commit_stmemory(self, state, action_values):
		self.stmemory.append({
			'state': state
			, 'AV': action_values
		})

	def commit_ltmemory(self):
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def clear_stmemory(self):
		self.stmemory = deque(maxlen=self.memory_size)
