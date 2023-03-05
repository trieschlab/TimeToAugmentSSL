import os
import random
import time
import numpy as np
import matplotlib
import torch
from torch.utils.data import DataLoader

from envs.six_objects import get_label_names
from tools.evaluation import lls, log_accuracies, saliency_map, confusion_matrix, get_pacmap, log_rewards_stats, \
    log_actions, log_replay_categories, log_views, RewLogger, knn_evaluation, get_all_representations, save_image, \
    eval_aggl
from tools.logger import EpochLogger
from tools.utils import TdwImageDataset, get_standard_dataset, get_test_dataset, get_normal_dataset, get_roll_dataset, \
    build_envname, preprocess


class DatasetHandler():
    
    def __init__(self, args, save_dir):
        self.args=args
        self.save_dir = save_dir
        self.datasets = []
        self.epoch_logger_symbolic = EpochLogger(output_dir=save_dir, exp_name="Seed-" + str(self.args.seed),output_fname="symbolic.txt")

        epoch_logger_valid = EpochLogger(output_dir=save_dir, exp_name="TestSeed-" + str(self.args.seed))
        epoch_logger_valid.save_config(self.args)

        self.datasets.append({"dataset":TdwImageDataset(self.args, "dataset.csv", get_standard_dataset(self.args)), "logger": epoch_logger_valid})
        ###Init environments and datasets
        self.test_dataset = None
        epoch_logger_test = EpochLogger(output_dir=save_dir, exp_name="Test2Seed-" + str(self.args.seed),
                                        output_fname='test_progress.txt')
        self.datasets.append({"dataset": TdwImageDataset(self.args,"dataset.csv" if not self.args.category else "dataset_test.csv",get_test_dataset(self.args)), "logger": epoch_logger_test})
        if self.args.full_logs:

            try:
                epoch_logger_no_trans2 = EpochLogger(output_dir=save_dir, exp_name="Test3Seed-" + str(self.args.seed),output_fname='progress2.txt')
                self.datasets.append({"dataset":TdwImageDataset(self.args,"dataset.csv", get_normal_dataset(self.args)),"logger": epoch_logger_no_trans2})

                epoch_logger_no_trans = EpochLogger(output_dir=save_dir, exp_name="Test3Seed-" + str(self.args.seed),output_fname='test2_progress.txt')
                self.datasets.append({"dataset":TdwImageDataset(self.args,"dataset_test.csv", get_normal_dataset(self.args)),"logger": epoch_logger_no_trans})
            except:print("test dataset not found")
            # if self.args.pitch:
            #     try:
            #         epoch_logger_pitch_test = EpochLogger(output_dir=save_dir, exp_name="Test3Seed-" + str(self.args.seed),output_fname='test_progress_pitch.txt')
            #         epoch_logger_pitch = EpochLogger(output_dir=save_dir, exp_name="Test3Seed-" + str(self.args.seed),output_fname='progress_pitch.txt')
            #         self.datasets.append({"dataset": TdwImageDataset(self.args,"dataset.csv", get_roll_dataset(self.args)),"logger": epoch_logger_pitch})
            #         self.datasets.append({"dataset": TdwImageDataset(self.args,"dataset.csv" if not self.args.category else "dataset_test.csv"),"logger": epoch_logger_pitch_test})
            #     except:print("Pitch dataset not found")
            if self.args.rot_logs and self.args.background < 10:
                try:
                    epoch_logger_rot = EpochLogger(output_dir=save_dir, exp_name="Test4Seed-" + str(self.args.seed),output_fname='rot_progress.txt')
                    epoch_logger_rot2 = EpochLogger(output_dir=save_dir, exp_name="Test5Seed-" + str(self.args.seed),output_fname='rot2_progress.txt')
                    epoch_logger_trot = EpochLogger(output_dir=save_dir, exp_name="Test4Seed-" + str(self.args.seed),output_fname='rot_train_progress.txt')
                    epoch_logger_trot2 = EpochLogger(output_dir=save_dir, exp_name="Test5Seed-" + str(self.args.seed),output_fname='rot2_train_progress.txt')
                    self.datasets.append({"dataset": TdwImageDataset(self.args, "dataset.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args, rotate=False) + "_r0_dataset"),"logger": epoch_logger_trot})
                    self.datasets.append({"dataset": TdwImageDataset(self.args,"dataset_test.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args,rotate=False) + "_r0_dataset"),"logger": epoch_logger_rot})
                    self.datasets.append({"dataset": TdwImageDataset(self.args, "dataset.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args, rotate=False) + "_r-90_dataset"),"logger": epoch_logger_trot2})
                    self.datasets.append({"dataset": TdwImageDataset(self.args,"dataset_test.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args,rotate=False) + "_r-90_dataset"),"logger": epoch_logger_rot2})

                except:print("Rotations dataset not found")
                try:
                    epoch_logger_rot3 = EpochLogger(output_dir=save_dir, exp_name="Test4Seed-" + str(self.args.seed),output_fname='rot3_progress.txt')
                    epoch_logger_trot3 = EpochLogger(output_dir=save_dir, exp_name="Test4Seed-" + str(self.args.seed),output_fname='rot3_train_progress.txt')
                    self.datasets.append({"dataset": TdwImageDataset(self.args, "dataset.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args, rotate=False) + "_r-80_dataset"),"logger": epoch_logger_trot3})
                    self.datasets.append({"dataset": TdwImageDataset(self.args,"dataset_test.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args,rotate=False) + "_r-80_dataset"),"logger": epoch_logger_rot3})
                except:print("Rotations -80 dataset not found")
                try:
                    epoch_logger_rot4 = EpochLogger(output_dir=save_dir, exp_name="Test4Seed-" + str(self.args.seed),output_fname='rot4_progress.txt')
                    epoch_logger_trot4 = EpochLogger(output_dir=save_dir, exp_name="Test4Seed-" + str(self.args.seed),output_fname='rot4_train_progress.txt')
                    self.datasets.append({"dataset": TdwImageDataset(self.args, "dataset.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args, rotate=False) + "_r-10_dataset"),"logger": epoch_logger_trot4})
                    self.datasets.append({"dataset": TdwImageDataset(self.args, "dataset_test.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args, rotate=False) + "_r-10_dataset"),"logger": epoch_logger_rot4})
                except:print("Rotations -10 dataset not found")

            if self.args.back_logs and self.args.background >= 10:
                try:
                    epoch_logger_rot = EpochLogger(output_dir=save_dir, exp_name="Test4Seed-" + str(self.args.seed),output_fname='back_progress.txt')
                    epoch_logger_trot = EpochLogger(output_dir=save_dir, exp_name="Test4Seed-" + str(self.args.seed),output_fname='back_train_progress.txt')
                    self.datasets.append({"dataset": TdwImageDataset(self.args,"dataset.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args,background=10 + (self.args.background + 10) % 40,rotate=False) + "_dataset"),"logger": epoch_logger_trot})
                    self.datasets.append({"dataset": TdwImageDataset(self.args,"dataset_test.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args,background=10 + (self.args.background + 10) % 40,rotate=False) + "_dataset"),"logger": epoch_logger_rot})

                    epoch_logger_rot2 = EpochLogger(output_dir=save_dir, exp_name="Test5Seed-" + str(self.args.seed),output_fname='back2_progress.txt')
                    epoch_logger_trot2 = EpochLogger(output_dir=save_dir, exp_name="Test5Seed-" + str(self.args.seed),output_fname='back2_train_progress.txt')

                    self.datasets.append({"dataset":TdwImageDataset(self.args,"dataset.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args,background=10 + (self.args.background + 20) % 40,rotate=False) + "_dataset"),"logger": epoch_logger_trot2})
                    self.datasets.append({"dataset":TdwImageDataset(self.args,"dataset_test.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args,background=10 + (self.args.background + 20) % 40,rotate=False) + "_dataset"),"logger": epoch_logger_rot2})
                except: print("Test2 and test3 dataset not found")
            if self.args.back_logs and self.args.background == 10:
                epoch_logger_one = EpochLogger(output_dir=save_dir, exp_name="Test5Seed-" + str(self.args.seed),output_fname='one_progress.txt')
                epoch_logger_tone = EpochLogger(output_dir=save_dir, exp_name="Test5Seed-" + str(self.args.seed),output_fname='one_train_progress.txt')

                self.datasets.append({"dataset": TdwImageDataset(self.args, "dataset.csv",os.environ["DATASETS_LOCATION"] + build_envname(self.args,rotate=False) + "_dataset_one"), "logger": epoch_logger_tone})
                self.datasets.append({"dataset": TdwImageDataset(self.args, "dataset_test.csv", os.environ["DATASETS_LOCATION"] + build_envname(self.args,rotate=False) + "_dataset_one"),"logger": epoch_logger_one})

    def agglomerative_evaluation(self, rep_function, proj=False):
        pre="proj_" if proj else ""
        v1_s_train, v2_s_train = eval_aggl(rep_function.args, rep_function.net, test=False,proj=proj)
        v1_s_test, v2_s_test = eval_aggl(rep_function.args, rep_function.net, test=True, proj=proj)
        self.epoch_logger_symbolic.log_tabular(pre+"train_acc_aggl_cat",v1_s_train)
        self.epoch_logger_symbolic.log_tabular(pre+"test_acc_aggl_cat",v1_s_test)
        self.epoch_logger_symbolic.log_tabular(pre+"train_acc_aggl_obj",v2_s_train)
        self.epoch_logger_symbolic.log_tabular( pre+"test_acc_aggl_obj",v2_s_test)

    def tests(self, rep_function, cpt_tests):
        try:
            if cpt_tests%3 == 0:
                self.agglomerative_evaluation(rep_function, proj=False)
                if self.args.projection:
                    self.agglomerative_evaluation(rep_function, proj=True)
                self.epoch_logger_symbolic.dump_tabular()
        except Exception as e:
            print(e)
            print("agglomerative not working")

        for i in range(len(self.datasets)):
            print_images=False
            store_acc = 0
            test_set=None
            if i %2 == 0:
                test_set = self.datasets[i+1]["dataset"]
            if i == 0:
                print_images = True
                store_acc = 1
            if i == 1:
                store_acc = 2
            if self.args.projection:
                self.test(self.datasets[i]["dataset"], rep_function, self.datasets[i]["logger"], cpt_tests,self.save_dir, print_images=print_images, store_acc=store_acc, test_set=test_set, proj=True)
            self.test(self.datasets[i]["dataset"], rep_function, self.datasets[i]["logger"], cpt_tests, self.save_dir, print_images=print_images, store_acc= store_acc, test_set=test_set)

    def save_reconstructions(self, images, features, decoder, save_dir, id):
        decoder.eval()
        for i in range(0,10):
            save_image(save_dir + "/reconstruction_"+str(id), i, images[i:i+1])
            image_recons = decoder(features[i:i+1])
            save_image(save_dir + "/reconstruction_"+str(id), str(i)+"_2", image_recons*255)
        decoder.train()

    def test(self, dataset, rep_function, logger, step_tests, save_dir, print_images= True, test_set=None, store_acc=0, proj=False):
        test_time = time.time()
        rep_function.net.eval()
        pre = "" if not proj else "proj_"

        features, labels, labels_background, labels_position, labels_category, images = get_all_representations(self.args,rep_function.net, dataset, proj=proj)

        #Object accuracy
        acc, sol, pred, acc5 = lls(features, labels, dataset.totalClasses)
        # wb = wcss_bcss(self.args, features, labels, dataset.totalClasses, save_dir=save_dir if print_images else None, step_tests=step_tests)
    
        #Background accuracy
        if self.args.back_logs:
            max_labels = torch.max(labels_background).item()+1
            acc_back, sol_b, _ , acc_back5 = lls(features, labels_background, max_labels)
            # wb_back = wcss_bcss(self.args, features, labels_background, max_labels)
    
        if self.args.category:
            labels_category = labels_category.to(self.args.device)
            max_labels_cat = torch.max(labels_category).item() + 1
            acc_cat, sol_cat, pred_cat, acc_cat5 = lls(features, labels_category, max_labels_cat)
            if store_acc:
                log_accuracies(save_dir, pred_cat.argmax(dim=-1), labels_category, max_labels_cat, "val_categories" if store_acc == 1 else "test_categories")
                log_accuracies(save_dir, pred.argmax(dim=-1), labels, dataset.totalClasses, "val_objects" if store_acc == 1 else "test_objects")
    
            # wb_cat = wcss_bcss(self.args, features, labels_category, max_labels_cat)
    
    
        if dataset.max_positions > 1 and self.args.back_logs:
            labels_positions = labels_position.to(self.args.device)
            max_labels_pos = torch.max(labels_positions).item() + 1
            acc_pos, sol_pos, _, acc_pos5 = lls(features, labels_positions, max_labels_pos)
            logger.log_tabular(pre+"final_acc_pos", acc_pos.item())
    
        # weight correlation
        # if print_images and step_tests != 0 and step_tests % 30 == 0 and proj and \
        if print_images and proj and ("recons" in self.args.regularizers or "pred_recons" in self.args.regularizers):
            self.save_reconstructions(images, features, rep_function.decoder, save_dir, step_tests//30)


        if not self.args.binocular and print_images and step_tests%30 == 0 and not proj:
            if self.args.full_logs:
                # idx = [i for i in range(0, images.shape[0], images.shape[0] // dataset.totalClasses)]
                idx = random.choices(range(0, images.shape[0]-1), k=5)
                saliency_map(images, idx, save_dir + "/saliency/" + str(step_tests) + "/", rep_function, features)
                # if not self.args.category:
                #     confusion_matrix(self.args, save_dir, pred, labels, dataset.totalClasses, step_tests, dataset)
                # else:
                #     confusion_matrix(self.args, save_dir, pred_cat, labels_category, max_labels_cat, step_tests, dataset)
                #
    
            # if not os.path.isdir(save_dir + "/correlation/"):
            #     os.makedirs(save_dir + "/correlation/")
            # # plot_importance(sol, sol_b, save_dir + "/correlation/" + str(step_tests) + ".png")
            # plot_importance(sol, sol_b, labels, labels_background, features, save_dir + "/correlation/" + str(step_tests) + ".png")
            # matplotlib.pyplot.close()
    
            #Clustering
            if self.args.pacmap:
                try:
                    if not self.args.category:
                        pacmap_plot, e = get_pacmap(features, labels, 0, dataset.totalClasses, get_label_names(self.self.args))
                    else:
                        pacmap_plot, e = get_pacmap(features, labels_category, 0, max_labels_cat, dataset.list_of_categories)
    
                    pacmap_plot.savefig(save_dir+"/clusters/output"+str(step_tests)+".png")
                    matplotlib.pyplot.close()
                except:
                    print("error during pacmap")
    
                if self.args.back_logs and self.args.background >= 10:
                    try:
                        pacmap_plot, _ = get_pacmap(features, labels_background, 0, dataset.max_backgrounds, [str(i) for i in range(dataset.max_backgrounds)], embedding=e)
                        pacmap_plot.savefig(save_dir + "/clusters_back/output" + str(step_tests) + ".png")
                        matplotlib.pyplot.close()
    
                        pacmap_plot, _ = get_pacmap(features, labels_positions, 0, dataset.max_positions, [str(i) for i in range(dataset.max_positions)], embedding=e)
                        pacmap_plot.savefig(save_dir + "/clusters_pos/output" + str(step_tests) + ".png")
                        matplotlib.pyplot.close()
                    except e:
                        print("error during pacmap", e)
    
        if test_set is not None and self.args.category:
            # test_dataloader = DataLoader(test_set, batch_size=len(test_set) + 1, shuffle=False)
            # images_test,_, _, _, labels_test = next(iter(test_dataloader))
            # with torch.no_grad():
            #     if images_test.shape[0] > 4000:
            #         features_test = torch.cat((rep_function.embed(images_test[:4000])[0 if not proj else 1], rep_function.embed(images_test[4000:])[0 if not proj else 1]),dim=0)
            #     else:
            #         features_test  = rep_function.embed(images_test)[0 if not proj else 1]
            features_test, _, _, _, labels_test, images_test = get_all_representations(self.args, rep_function.net, test_set, proj=proj)

            labels_test = labels_test.to(self.args.device)
            # pred_test = torch.mm(features_test, sol)
            prediction_test = features_test @ sol_cat
            hard_prediction_test = prediction_test.argmax(dim=-1)
    
            acc_cat_test = (hard_prediction_test == labels_test).sum() / len(features_test)
            five_predictions_test = torch.topk(prediction_test, min(5,prediction_test.shape[1]), dim=-1).indices
            acc_cat5_test = (labels_test.unsqueeze(-1) == five_predictions_test).sum() / len(features_test)
            logger.log_tabular(pre+"final_acc_cat_test", acc_cat_test.item())
            logger.log_tabular(pre+"final_acc_cat5_test", acc_cat5_test.item())
            if self.args.knn_logs:
                acc_cat_knn = knn_evaluation(features, labels_category, features_test, labels_test, max_labels_cat, rep_function)
                logger.log_tabular(pre+"acc_cat_knn", acc_cat_knn.item())

        all_time = time.time() - test_time
        rep_function.net.train()
        if self.args.category:
            logger.log_tabular(pre+"final_acc_cat", acc_cat.item())
            logger.log_tabular(pre+"final_acc_cat5", acc_cat5.item())
    
            # logger.log_tabular("wb_cat", wb_cat.item())
    
        logger.log_tabular(pre+"final_acc", acc.item())
        logger.log_tabular(pre+"final_acc5", acc5.item())
        # logger.log_tabular("wb", wb.item())
        if self.args.back_logs:
            logger.log_tabular(pre+"final_acc_back", acc_back.item())
            logger.log_tabular(pre+"final_acc_back5", acc_back5.item())
            # logger.log_tabular("wb_back", wb_back.item())
        if not proj:
            logger.log_tabular("epoch", step_tests)
            logger.log_tabular("all_time", all_time)
            logger.log_tabular("timesteps", self.args.num_updates*step_tests)
            logger.dump_tabular()

class LogTrainer():
    def __init__(self, args, agent, rep_function, buffer, save_dir):
        self.args=args
        self.epoch_logger_train = EpochLogger(output_dir=save_dir, exp_name="TrainSeed-" + str(args.seed),output_fname='train_progress.txt')
        self.epoch_logger_longtrain = EpochLogger(output_dir=save_dir, exp_name="LongTrainSeed-" + str(args.seed),output_fname='long_train_progress.txt')
        self.log_objects, self.log_backgrounds, self.fix_objects, self.fix_backgrounds, self.aggr_actions = None,None, None,None, []
        self.save_dir =save_dir
        self.agent=agent
        self.rep_function = rep_function
        self.buffer=buffer
        self.rew_logger=None
        self.train_timer = time.time()

    def update_objects_backgrounds(self,log_objects, log_backgrounds, fix_objects, fix_backgrounds, infos, args):
        keylog = "total_num_categories" if args.category else "total_num_objects"
        cptlog = "category" if args.category else "oid"

        if log_objects is None:
            log_objects = np.zeros(infos[keylog], dtype=np.int)
            log_backgrounds = np.zeros(infos["total_num_positions"], dtype=np.int)
            fix_objects = np.zeros(infos[keylog], dtype=np.int)
            fix_backgrounds = np.zeros(infos["total_num_positions"], dtype=np.int)
        log_objects[infos[cptlog]] += 1
        log_backgrounds[infos["position"]] += 1
        # print(infos["fix"], infos[cptlog])
        if infos["fix"]:
            fix_objects[infos[cptlog]] += 1
            fix_backgrounds[infos["position"]] += 1

        return log_objects, log_backgrounds, fix_objects, fix_backgrounds

    def update(self, info, a, sample):
        self.log_objects, self.log_backgrounds, self.fix_objects, self.fix_backgrounds = self.update_objects_backgrounds(self.log_objects,
                                                                                            self.log_backgrounds,
                                                                                            self.fix_objects,
                                                                                            self.fix_backgrounds, info, self.args)
        self.aggr_actions.append(a.cpu().numpy())
        if self.rew_logger is None: self.rew_logger = RewLogger(self.args, info, self.save_dir, self.rep_function)
        self.rew_logger.update_stats(sample)



    def log(self, time_for_step, cpt_train, info, sample):
        # if args.epochs_type == 0:
        # else:
        #     sample = train_epoch(agent, rep_function, buffer, rew_logger, cpt_train)

        if not cpt_train % self.args.log_interval:
            self.agent.log(self.epoch_logger_train)
            self.rep_function.log(self.epoch_logger_train)
            self.epoch_logger_train.log_tabular("fps", self.args.log_interval / (time.time() - self.train_timer))
            self.epoch_logger_train.log_tabular("steptime", time_for_step / self.args.log_interval)
            self.epoch_logger_train.log_tabular("timesteps", cpt_train)
            self.epoch_logger_train.dump_tabular()
            self.train_timer = time.time()
            time_for_step = 0
        if not cpt_train % (5 * self.args.log_interval):
            new = cpt_train == 5 * self.args.log_interval
            if self.args.epochs_number:
                sample = self.buffer.sample()
                self.rep_function.learn(sample, compute_rewards=False)
            self.log_objects, self.log_backgrounds, self.fix_objects, self.fix_backgrounds = self.log_objects_backgrounds(self.log_objects,
                                                                                                 self.log_backgrounds,
                                                                                                 self.fix_objects,
                                                                                                 self.fix_backgrounds,
                                                                                                 self.save_dir, new)
            if self.args.log_backgrounds_probs:
                log_rewards_stats(self.rep_function, info["total_num_positions"], self.save_dir, sample)
                self.rew_logger.log(self.buffer)
                # log_rewards_obj_stats(rep_function, info["total_num_objects"], save_dir, new)
            self.epoch_logger_longtrain.log_tabular("timesteps", cpt_train)
            self.aggr_actions = log_actions(self.aggr_actions, self.epoch_logger_longtrain)
            if self.args.category: log_replay_categories(self.buffer, info["total_num_categories"], info["total_num_objects"],self.save_dir)
            log_views(self.buffer, self.epoch_logger_longtrain)
            self.epoch_logger_longtrain.dump_tabular()
        return time_for_step

    def log_objects_backgrounds(self,log_objects, log_backgrounds, fix_objects, fix_backgrounds, save_dir, new):
        import csv
        with open(save_dir+"/objects.csv", 'a') as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["o"+str(i) for i in range(log_objects.shape[0])])
            writer.writerow(log_objects.tolist())
        with open(save_dir+"/backgrounds.csv", 'a') as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["pos"+str(i) for i in range(log_backgrounds.shape[0])])
            writer.writerow(log_backgrounds.tolist())
        with open(save_dir+"/fix_objects.csv", 'a') as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["o"+str(i) for i in range(log_objects.shape[0])])
            writer.writerow((fix_objects.astype(np.float32)/(log_objects.astype(np.float32)+0.01)).tolist())
        with open(save_dir+"/fix_backgrounds.csv", 'a') as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["pos"+str(i) for i in range(log_backgrounds.shape[0])])
            writer.writerow((fix_backgrounds.astype(np.float32)/(log_backgrounds.astype(np.float32)+0.01)).tolist())
        log_objects.fill(0)
        log_backgrounds.fill(0)
        fix_objects.fill(0)
        fix_backgrounds.fill(0)

        return log_objects, log_backgrounds, fix_objects, fix_backgrounds