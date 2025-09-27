import json
with open('/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/blp.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and len(cell['source']) > 10:
        source = ''.join(cell['source'])
        # replace utility equation
        old_eq = """uijt=(1)xjt+(2)
isatellite jt+(3)
iwired jt+pjt+t+
ijtj >0
ui0t=
i0t;"""
        new_eq = """$u_{ijt} = \\beta_1 x_{jt} + \\beta_2 i_{satellite,jt} + \\beta_3 i_{wired,jt} + \\alpha p_{jt} + \\xi_{jt} + \\epsilon_{ijt}$ for $j > 0$
$u_{i0t} = \\epsilon_{i0t}$"""
        source = source.replace(old_eq, new_eq)
        # replace cost
        old_cost = """lnmcjt=
0+wjt
1+!jt=8;"""
        new_cost = """$\\ln mc_{jt} = \\gamma_0 + w_{jt} \\gamma_1 + \\omega_{jt}$"""
        source = source.replace(old_cost, new_cost)
        # replace FOC
        old_foc = """pjtmcjt@ s jt(pt)
@pjt+sjt= 0
cat > format_notebook.py << 'EOF'
import json
with open('/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/blp.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and len(cell['source']) > 10:
        source = ''.join(cell['source'])
        # replace utility equation
        old_eq = """uijt=(1)xjt+(2)
isatellite jt+(3)
iwired jt+pjt+t+
ijtj >0
ui0t=
i0t;"""
        new_eq = """$u_{ijt} = \\beta_1 x_{jt} + \\beta_2 i_{satellite,jt} + \\beta_3 i_{wired,jt} + \\alpha p_{jt} + \\xi_{jt} + \\epsilon_{ijt}$ for $j > 0$
$u_{i0t} = \\epsilon_{i0t}$"""
        source = source.replace(old_eq, new_eq)
        # replace cost
        old_cost = """lnmcjt=
0+wjt
1+!jt=8;"""
        new_cost = """$\\ln mc_{jt} = \\gamma_0 + w_{jt} \\gamma_1 + \\omega_{jt}$"""
        source = source.replace(old_cost, new_cost)
        # replace FOC
        old_foc = """pjtmcjt@ s jt(pt)
@pjt+sjt= 0

@pjt1
sjt (1)"""
        new_foc = """$(p_{jt} - mc_{jt}) \\frac{\\partial s_{jt}(p_t)}{\\partial p_{jt}} + s_{jt} = 0$
$\\Rightarrow p_{jt} - mc_{jt} = - \\left( \\frac{\\partial s_{jt}(p_t)}{\\partial p_{jt}} \\right)^{-1} s_{jt}$"""
        source = source.replace(old_foc, new_foc)
        # replace data params
        old_data = """(1)= 1 ,(k)
iidN(4;1)fork= 2;3
=2

(0)= 1=2,
(1)= 1=4:"""
        new_data = """$\\beta_1 = 1, \\beta_k \\sim iid N(4,1)$ for $k=2,3$
$\\alpha = -2$
$\\gamma_0 = 1/2, \\gamma_1 = 1/4$"""
        source = source.replace(old_data, new_data)
        cell['source'] = source.split('\n')
with open('/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/blp.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('Notebook formatted with LaTeX equations.')
