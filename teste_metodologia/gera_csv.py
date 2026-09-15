import cv2
import pandas as pd

# 1. Map dos dados brutos com intervalos por vídeo (chaves numéricas: 1, 2, 3, 4)
intervals_data = {
    1: [
        (0.0, 210.0, 'alert'),           # 00:00 - 03:30
        (211.0, 240.0, 'fatigue'),       # 03:31 - 04:00
        (241.0, 317.0, 'distraction'),   # 04:01 - 05:17
        (318.0, 333.0, 'fatigue'),       # 05:18 - 05:33
        (334.0, 349.0, 'alert'),         # 05:34 - 05:49
        (350.0, 588.0, 'distraction'),   # 05:50 - 09:48
        (589.0, 979.0, 'alert'),         # 09:49 - 16:19
        (980.0, 990.0, 'fatigue'),       # 16:20 - 16:30
        (991.0, 1106.0, 'alert'),        # 16:31 - 18:26
        (1107.0, 1433.0, 'absent'),      # 18:27 - 23:53
        (1434.0, 1463.0, 'alert'),       # 23:54 - 24:23
        (1465.0, 1637.0, 'absent'),      # 24:25 - 27:17
        (1638.0, 1776.0, 'alert')        # 27:18 - 29:36
    ],
    2: [
        (0.0, 150.0, 'alert'),           # 00:00 - 02:30
        (151.0, 152.0, 'distraction'),   # 02:31 - 02:32
        (152.0, 154.0, 'alert'),         # 02:32 - 02:34
        (155.0, 159.0, 'distraction'),   # 02:35 - 02:39
        (160.0, 512.0, 'alert'),         # 02:40 - 08:32
        (513.0, 519.0, 'distraction'),   # 08:33 - 08:39
        (520.0, 780.0, 'alert'),         # 08:40 - 13:00
        (781.0, 894.0, 'distraction'),   # 13:01 - 14:54
        (895.0, 1247.0, 'alert'),        # 14:55 - 20:47
        (1248.0, 1320.0, 'distraction'), # 20:48 - 22:00
        (1321.0, 1499.0, 'alert'),       # 22:01 - 24:59
        (1500.0, 1537.0, 'distraction'), # 25:00 - 25:37
        (1538.0, 1644.0, 'alert'),       # 25:38 - 27:24
        (1645.0, 1651.0, 'distraction'), # 27:25 - 27:31
        (1652.0, 1694.0, 'alert'),       # 27:32 - 28:14
        (1695.0, 1732.0, 'distraction'), # 28:15 - 28:52
        (1733.0, 1794.0, 'alert')        # 28:53 - 29:54
    ],
    3: [
        (0.0, 456.0, 'alert'),           # 00:00 - 07:36
        (457.0, 899.0, 'absent'),        # 07:37 - 14:59
        (900.0, 1063.0, 'alert'),        # 15:00 - 17:43
        (1064.0, 1069.0, 'distraction'), # 17:44 - 17:49
        (1070.0, 1174.0, 'alert'),       # 17:50 - 19:34
        (1175.0, 1198.0, 'distraction'), # 19:35 - 19:58
        (1199.0, 1239.0, 'alert'),       # 19:59 - 20:39
        (1240.0, 1254.0, 'distraction'), # 20:40 - 20:54
        (1255.0, 1470.0, 'alert'),       # 20:55 - 24:30
        (1471.0, 1480.0, 'distraction'), # 24:31 - 24:40
        (1481.0, 1556.0, 'alert'),       # 24:41 - 25:56
        (1557.0, 1558.0, 'distraction'), # 25:57 - 25:58
        (1559.0, 1651.0, 'alert'),       # 25:59 - 27:31
        (1652.0, 1653.0, 'fatigue'),     # 27:32 - 27:33
        (1654.0, 1673.0, 'alert'),       # 27:34 - 27:53
        (1674.0, 1680.0, 'fatigue'),     # 27:54 - 28:00
        (1681.0, 1689.0, 'alert'),       # 28:01 - 28:09
        (1690.0, 1697.0, 'fatigue'),     # 28:10 - 28:17
        (1698.0, 1699.0, 'alert'),       # 28:18 - 28:19
        (1700.0, 1705.0, 'fatigue'),     # 28:20 - 28:25
        (1706.0, 1735.0, 'distraction'), # 28:26 - 28:55
        (1736.0, 1752.0, 'alert'),       # 28:56 - 29:12
        (1753.0, 1770.0, 'distraction'), # 29:13 - 29:30
        (1771.0, 1784.0, 'alert')        # 29:31 - 29:44
    ],
    4: [
        (0.0, 22.0, 'alert'),            # 00:00 - 00:22
        (23.0, 55.0, 'distraction'),     # 00:23 - 00:55
        (56.0, 84.0, 'alert'),           # 00:56 - 01:24
        (85.0, 935.0, 'absent'),         # 01:25 - 15:35
        (936.0, 1052.0, 'alert'),        # 15:36 - 17:32
        (1053.0, 1061.0, 'distraction'), # 17:33 - 17:41
        (1062.0, 1165.0, 'alert'),       # 17:42 - 19:25
        (1166.0, 1172.0, 'distraction'), # 19:26 - 19:32
        (1173.0, 1174.0, 'fatigue'),     # 19:33 - 19:34
        (1175.0, 1178.0, 'alert'),       # 19:35 - 19:38
        (1179.0, 1180.0, 'fatigue'),     # 19:39 - 19:40
        (1181.0, 1183.0, 'distraction'), # 19:41 - 19:43
        (1184.0, 1262.0, 'alert'),       # 19:44 - 21:02
        (1263.0, 1274.0, 'distraction'), # 21:03 - 21:14
        (1275.0, 1503.0, 'alert'),       # 21:15 - 25:03
        (1504.0, 1559.0, 'distraction'), # 25:04 - 25:59
        (1560.0, 1785.0, 'alert')        # 26:00 - 29:45
    ]
}

# 2. Mapeamento dos vídeos utilizando chave numérica para bater com o intervals_data
video_files = {
    1: '/home/eduardo/Code/distraction_detection/fase_2/data/raw/1.mp4',
    2: '/home/eduardo/Code/distraction_detection/fase_2/data/raw/2.mp4',
    3: '/home/eduardo/Code/distraction_detection/fase_2/data/raw/3.mp4',
    4: '/home/eduardo/Code/distraction_detection/fase_2/data/raw/4.mp4'
}

# 3. Defina aqui onde deseja salvar o CSV
output_csv_path = '/home/eduardo/Code/distraction_detection/teste_metodologia/results/classificacoes_frames_exatos.csv'

def get_state(timestamp, intervals):
    for start, end, state in intervals:
        if start <= timestamp <= end:
            return state
    return 'unknown'

rows = []

for vid_num, file_path in video_files.items():
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {file_path}")
        continue
    
    video_id = f"video_{vid_num:02d}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Lendo {video_id} | FPS detectado: {fps}")

    frame_index = 0
    intervals = intervals_data.get(vid_num, [])

    while True:
        # grab() avança o ponteiro do frame sem decodificar a matriz de imagem
        ret = cap.grab()
        if not ret:
            break

        timestamp_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        if timestamp_seconds == 0 and frame_index > 0:
            timestamp_seconds = frame_index / fps

        state = get_state(timestamp_seconds, intervals)

        rows.append({
            'video_id': video_id,
            'frame_index': frame_index,
            'timestamp_seconds': round(timestamp_seconds, 6),
            'state': state
        })

        frame_index += 1

    cap.release()

# 4. Salvar CSV no caminho especificado
df = pd.DataFrame(rows)
df.to_csv(output_csv_path, index=False)
print(f"CSV gerado com sucesso em: {output_csv_path}")