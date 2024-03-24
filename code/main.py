import datetime
import torch
import numpy as np
from utils import load_data
from evaluation import score
from args import read_args
from model import MILHTI
from torch.utils.data import DataLoader, TensorDataset
from loss_function import FocalLoss



def evaluate(model, data_loader, loss_func, loss_fcn_ins, para1, para2):
    model.eval()

    logits_total = []
    labels_total = []
    logits_total2 = []
    labels_total2 = []
    loss1 = 0
    batch_sum = 0

    with torch.no_grad():
        for batch_idx, (herb, target, ingredient, label) in enumerate(data_loader):
            logits, logits2, loss_mse = model(herb, target, ingredient)
            logits_total.append(logits)
            labels_total.append(label)
            logits_total2.append(logits2)
            labels_total2.append(label)
            loss1 = loss1 + loss_mse
            batch_sum = batch_sum + 1

    logits_total = torch.cat(logits_total, dim=0)
    labels_total = torch.cat(labels_total, dim=0)
    loss = loss_func(logits_total, labels_total)
    logits_total2 = torch.cat(logits_total2, dim=0)
    labels_total2 = torch.cat(labels_total2, dim=0)
    loss2 = loss_fcn_ins(logits_total2, labels_total2)

    accuracy, auroc, auprc, f1, precision, recall = score(logits_total, labels_total)
    return loss + para2 * (loss1 / batch_sum) + para1 * loss2, accuracy, auroc, auprc, f1, precision, recall


def milhti_main(args):
    dt = datetime.datetime.now()
    post_fix = "{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )

    print("load data...")
    (herb_init, target_init, ingredient_init, instance_idx, labels, train_idx, val_idx, test_idx) = load_data()

    herb_init = torch.tensor(herb_init)
    herb_init = herb_init.to(args["device"])
    target_init = torch.tensor(target_init)
    target_init = target_init.to(torch.long)
    target_init = target_init.to(args["device"])
    ingredient = []
    for i in range(len(instance_idx)):
        temp = []
        padded_array = np.zeros((76, len(ingredient_init[0])))
        for j in range(len(instance_idx[i])):
            temp.append(ingredient_init[instance_idx[i][j]])
        temp = np.array(temp)
        padded_array[:temp.shape[0], :] = temp
        padded_array = torch.tensor(padded_array)
        ingredient.append(padded_array)
    ingredient = torch.stack(ingredient)
    ingredient = ingredient.to(args["device"])
    labels = torch.tensor(labels)
    labels = labels.to(args["device"])
    labels = labels.float()
    labels = torch.unsqueeze(labels, 1)
    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    test_idx = torch.tensor(test_idx)

    num = len(target_init)
    train_h_id = [int(idx / num) for idx in train_idx]
    val_h_id = [int(idx / num) for idx in val_idx]
    test_h_id = [int(idx / num) for idx in test_idx]
    train_t_id = [int(idx % num) for idx in train_idx]
    val_t_id = [int(idx % num) for idx in val_idx]
    test_t_id = [int(idx % num) for idx in test_idx]

    data_train = TensorDataset(herb_init[train_h_id], target_init[train_t_id], ingredient[train_h_id], labels[train_idx])
    data_val = TensorDataset(herb_init[val_h_id], target_init[val_t_id], ingredient[val_h_id], labels[val_idx])
    data_test = TensorDataset(herb_init[test_h_id], target_init[test_t_id], ingredient[test_h_id], labels[test_idx])

    train_loader = DataLoader(dataset=data_train, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=data_val, batch_size=args["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=data_test, batch_size=args["batch_size"], shuffle=True)

    print("data loaded successfully")
    print("-----------------------------------------------train-----------------------------------------------")

    model = MILHTI(
        herb_input_dim=args["herb_input_dim"],
        herb_output_dim=args["herb_output_dim"],
        target_input_dim=args["target_input_dim"],
        target_output_dim=args["target_output_dim"],
        ingredient_input_dim=args["ingredient_input_dim"],
        ingredient_output_dim=args["ingredient_output_dim"],
        drop_out=args["dropout"]
    ).to(args["device"])
    para1 = args["lambda1"]
    para2 = args["lambda2"]

    loss_fcn_ins = FocalLoss()
    loss_fcn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    best_f1 = 0.1
    for epoch in range(args["num_epochs"]):
        model.train()
        for batch_idx, (herb, target, ingredient, label) in enumerate(train_loader):
            logits, logits2, loss_mse = model(herb, target, ingredient)
            loss = loss_fcn(logits, label) + para1 * loss_fcn_ins(logits2, label) + para2 * loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_acc, val_auroc, val_auprc, val_f1, val_precision, val_recall = evaluate(
            model, val_loader, loss_fcn, loss_fcn_ins, para1, para2
        )
        print(
            "Epoch {:d} | Val Loss {:.4f} | Val AUROC {:.4f} | Val AUPRC {:.4f} | Val F1 {:.4f} | Val P {:.4f} | Val "
            "R {:.4f}".format(epoch, val_loss.item(), val_auroc, val_auprc, val_f1, val_precision, val_recall)
        )
        if best_f1 < val_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'result/best_model_{}.pth'.format(post_fix))
    print("-----------------------------------------------test-----------------------------------------------")
    best_model = model
    best_model.load_state_dict(torch.load('result/best_model_{}.pth'.format(post_fix)))
    test_loss, test_acc, test_auroc, test_auprc, test_f1, test_precision, test_recall = evaluate(
        best_model, test_loader, loss_fcn, loss_fcn_ins, para1, para2
    )
    print(
        "Test loss {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f} | Test F1 {:.4f} | Test P {:.4f} | "
        "Test R {:.4f}".format(test_loss.item(), test_auroc, test_auprc, test_f1, test_precision, test_recall)
    )


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print("start time: ", start_time_str)
    print("******************************************************************************************************")

    args = read_args()
    milhti_main(args)

    end_time = datetime.datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("******************************************************************************************************")
    print("end time: ", end_time_str)
    duration = end_time - start_time
    duration_seconds = duration.total_seconds()
    print("running :", duration_seconds)


