"""Run a command template in parallel over all combinations of arguments.

Usage examples:

  # Single variable
  python parallel_run.py "python script.py {f}" --f a.tdms b.tdms c.tdms

  # Cartesian product of two variables
  python parallel_run.py "python script.py {f} --harmonic {h}" \
    --f a.tdms b.tdms --h 1 2 3

  # Max parallelism (default: number of CPUs)
  python parallel_run.py -j 3 "python script.py {f}" --f a.tdms b.tdms c.tdms
"""

import argparse
import itertools
import subprocess
import sys
import time


def main():
    # Split argv: everything before first --key is for us, rest is variable defs
    template = None
    max_workers = None
    var_args = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-j" and i + 1 < len(sys.argv):
            max_workers = int(sys.argv[i + 1])
            i += 2
        elif template is None and not arg.startswith("--"):
            template = arg
            i += 1
        else:
            var_args.append(arg)
            i += 1

    if template is None:
        print(__doc__)
        sys.exit(1)

    # Parse variable definitions: --name val1 val2 ... --name2 val1 ...
    variables = {}
    current_key = None
    for arg in var_args:
        if arg.startswith("--"):
            current_key = arg[2:]
            variables[current_key] = []
        elif current_key is not None:
            variables[current_key].append(arg)

    if not variables:
        print("No variables defined. Use --name val1 val2 ...")
        sys.exit(1)

    # Generate all combinations
    keys = list(variables.keys())
    values = [variables[k] for k in keys]
    combos = list(itertools.product(*values))

    print(f"Template: {template}")
    print(f"Variables: {', '.join(f'{k}({len(variables[k])})' for k in keys)}")
    print(f"Jobs: {len(combos)}, workers: {max_workers or 'all'}")
    print()

    # Build commands
    jobs = []
    for combo in combos:
        cmd = template
        label_parts = []
        for k, v in zip(keys, combo):
            cmd = cmd.replace(f"{{{k}}}", v)
            label_parts.append(f"{k}={v}")
        jobs.append((cmd, ", ".join(label_parts)))

    # Run in parallel
    t0 = time.time()
    active = {}
    done = 0
    failed = 0
    queue = list(jobs)

    while queue or active:
        # Launch jobs up to max_workers
        while queue and (max_workers is None or len(active) < max_workers):
            cmd, label = queue.pop(0)
            # Fix forward slashes in executable path for cmd.exe on Windows
            if sys.platform == "win32":
                parts = cmd.split(None, 1)
                parts[0] = parts[0].replace("/", "\\")
                cmd = " ".join(parts)
            p = subprocess.Popen(cmd, shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            active[p] = (cmd, label, time.time())
            print(f"  START: {label}")

        # Check for completed
        for p in list(active):
            ret = p.poll()
            if ret is not None:
                cmd, label, t_start = active.pop(p)
                elapsed = time.time() - t_start
                done += 1
                if ret == 0:
                    print(f"  DONE:  {label} ({elapsed:.0f}s)")
                else:
                    failed += 1
                    output = p.stdout.read().decode(errors="replace")
                    print(f"  FAIL:  {label} (exit {ret}, {elapsed:.0f}s)")
                    for line in output.strip().split("\n")[-5:]:
                        print(f"         {line}")

        if active:
            time.sleep(0.5)

    elapsed = time.time() - t0
    print(f"\n=== {done} jobs in {elapsed:.0f}s ({failed} failed) ===")


if __name__ == "__main__":
    main()
