import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Function to read and visualize IMU data from a CSV file
def visualize_imu_data(path):
    time_format = '%Y-%m-%d %H:%M:%S.%f'

    data = pd.read_csv(path,
                       index_col='time',
                       parse_dates=['time'],
                       date_format=time_format,
                       dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['x'], label='x', color='r')
    plt.xlabel('Time')
    plt.ylabel('Acceleration x')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['y'], label='y', color='g')
    plt.xlabel('Time')
    plt.ylabel('Acceleration y')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.subplot(3, 1, 3)
    plt.plot(data.index, data['z'], label='z', color='b')
    plt.xlabel('Time')
    plt.ylabel('Acceleration z')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.show()


csv_file = './data/P001.csv.gz'
visualize_imu_data(csv_file)