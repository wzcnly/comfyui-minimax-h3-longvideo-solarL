"""Add optional schedule socket and safe default cache widget to existing graphs."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]/'examples'
for suffix in ('', '_SelfLift'):
    graph = json.loads((root/f'2080ti_内部循环_终版1{suffix}.json').read_text(encoding='utf-8'))
    node = next(n for n in graph['nodes'] if n['type'] in ('MiniMaxH3LongVideoSampler-solarL', 'MiniMaxH3SelfLiftLongVideoSampler-solarL'))
    node_id = max(n['id'] for n in graph['nodes'])+1
    edge_id = max(e[0] for e in graph['links'])+1
    target_slot = len(node['inputs'])
    node['inputs'].append({'name':'segment_plan','type':'SOLARL_H3_SEGMENT_PLAN','link':edge_id})
    node['widgets_values'][0:2] = [42, 'fixed']
    node.setdefault('widgets_values', []).append('')
    node.setdefault('widgets_values_named', {})['cache_directory'] = ''
    node['widgets_values_named'].update(seed=42, control_after_generate='fixed')
    for media in graph['nodes']:
        if media['type'] == 'VHS_LoadVideo-solarL' and isinstance(media.get('widgets_values'), dict):
            media['widgets_values']['video'] = ''
            media['widgets_values'].pop('videopreview', None)
    graph['nodes'].append({'id':node_id,'type':'MiniMaxH3SegmentPlan-solarL','pos':[node['pos'][0]-430,node['pos'][1]],
        'size':[400,330],'flags':{},'order':0,'mode':0,'inputs':[],
        'outputs':[{'name':'segment_plan','type':'SOLARL_H3_SEGMENT_PLAN','links':[edge_id]},
                   {'name':'report','type':'STRING','links':[]}],
        'widgets_values':['[{"frames":124,"prompt":""},{"frames":102,"prompt":""},{"frames":77,"prompt":""}]',22,''],
        'properties':{'Node name for S&R':'MiniMaxH3SegmentPlan-solarL'}})
    graph['links'].append([edge_id,node_id,0,node['id'],target_slot,'SOLARL_H3_SEGMENT_PLAN'])
    graph.update(last_node_id=node_id,last_link_id=edge_id)
    graph.pop('id',None)
    (root/f'2080ti_分段计划与缓存{suffix}.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Generated schedule example: {suffix or "ordinary"}')
