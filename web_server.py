import gradio as gr
import random
import numpy as np
# import pandas as pd
import csv

from ansamble import Ansamble


black_box = Ansamble()

# OURS 0: KLIKUN , 1: MALIY, 2: shipun
# ORGS MALIY: 1, KLIKUN: 2, SHIPUN: 3
transition_dict = {
    0: 2,
    1: 1,
    2: 3
}


def neural_network(images):
    result = [[img, np.array(
        [random.random(), random.random(), random.random()])] for img in images]
    for i in result:
        i[1] /= np.sum(i[1])
    return result


def create_csv(data, full_type=False):
    with open(f'results{"_full" if full_type else ""}.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["name", "class"] + full_type *
                        ['class_0', 'class_1', 'class_2'])
        for i in data:
            id = transition_dict[np.argmax(i[1])]

            writer.writerow([str(i[0].split('/')[-1]), str(id)
                             ] + full_type*[i[1][0], i[1][1], i[1][2]])


def callback_function(files):
    lables = ['КЛИКУН', 'МАЛЫЙ', 'ШИПУН']
    results = black_box.multiple_inference(
        [i.name.replace("\\", "/") for i in files])

    # return ['a', 'b', [(cv.imread(i.name), str(i.name)[-5:]) for i in files]]
    resulting_gallery = [(i[0], lables[np.argmax(
        i[1])] + f' {int(np.max(i[1])*100)}% ' + str(i[0].split('/')[-1].split('.')[0])) for i in results]

    create_csv(results)
    create_csv(results, True)

    # class_amount = [0,0,0]
    # for i in results:
    #     class_amount[np.argmax(i[1])]+=1
    # simple = pd.DataFrame({
    #     "class": ["0", "1", "2"],
    #     "amount": class_amount
    # })
    return ['results.csv', 'results_full.csv', resulting_gallery]


gallery = gr.Gallery(label="Images", show_label=True)
gallery.style(grid=(5, 5))
demo = gr.Interface(
    callback_function,
    gr.File(file_count="multiple", file_types=['image']),
    [gr.File(label="CSV table with results", show_label=True),
     gr.File(label="CSV table with extended results", show_label=True),
     #  gr.BarPlot(x='class', y='amount', title="total stats", label="Dataset statistics", show_label=True, width=300),
     gallery],
    allow_flagging='never'
)

if __name__ == "__main__":
    demo.launch(share=True)
