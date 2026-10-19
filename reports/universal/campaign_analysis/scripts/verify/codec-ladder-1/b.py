exec(open('a.py').read().split("print(len(units))")[0])
exec("\n".join(open('a.py').read().split("print(len(units))")[1].split("\n")[3:]).split("for e in [0,0.002]:\n    w,R")[0])
