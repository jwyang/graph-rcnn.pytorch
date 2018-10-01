import multiprocessing

class DataRunnerMP:
    """
    A multi-processing data runner for tensorflow
    """
    def __init__(self, task_func, task_generator, input_pls, capacity=100):
        self._input_pls = input_pls
        self._task_func = task_func
        self._task_generator = task_generator
        self.counter = 0
        self.processes = []

        self._queue_outputs = self._input_pls
        self.capacity = capacity

    def get_feed_batch(self):
        if self.counter % 100 == 0:
            print('qlen=%i' % self.data_queue.qsize())
        self.counter += 1
        feed = self.data_queue.get()
        out_feed= {}
        for k, v in feed.items():
            out_feed[self._input_pls[k]] = v
        return out_feed

    def get_inputs(self):
        return dict(self._queue_outputs)

    def _worker_main(self, task_queue, data_queue):
        """
        generate sample from task queue and put the sample
        into a data queue in the form of tf feed_dict
        """
        while True:
            task = task_queue.get()
            sample = self._task_func(task)
            if sample is None:
                continue
            feed = {}
            for key, pl in self._input_pls.items():
                feed[key] = sample[key]
            data_queue.put(feed)

    def _manager_main(self, queue):
        """
        put tasks into queue
        """
        for task in self._task_generator():
            queue.put(task)

    def start_processes(self, sess, n_processes=1):
        self.task_queue = multiprocessing.Queue(self.capacity)
        self.data_queue = multiprocessing.Queue(self.capacity)
        p = multiprocessing.Process(target=self._manager_main, args=(self.task_queue,))
        p.daemon = True
        p.start()
        self.processes.append(p)
        for n in range(n_processes):
            p = multiprocessing.Process(target=self._worker_main, args=(self.task_queue,self.data_queue))
            p.daemon = True
            p.start()
            self.processes.append(p)
