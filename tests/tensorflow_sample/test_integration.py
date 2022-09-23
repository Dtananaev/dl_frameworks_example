"""Integration test.

Note:
    During training tensorflow grabs most of GPU memory, threfore each process should include tensorflow import.
"""
__copyright__ = """
Copyright (c) 2022 Tananaev Denis
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import multiprocessing
import os

def _run_training(output_dir:str)->None:
    """Run training."""
    from framework_examples.tensorflow_sample.train import train
    from framework_examples.parameters import Parameters

    test_args ={
        "summary_dir":os.path.join(output_dir, "outputs"),
        "checkpoint_dir": os.path.join(output_dir, "checkpoint"),
        "epochs": 1,
        "batchsize": 1,
    }

    param = Parameters(**test_args)
    train(parameters=param)

def test_integration(tmp_dir: str)-> None:
    """Integration test."""
    training_process = multiprocessing.Process(target=_run_training, args=(tmp_dir,))
    training_process.start()
    training_process.join()
    assert training_process.exitcode ==0