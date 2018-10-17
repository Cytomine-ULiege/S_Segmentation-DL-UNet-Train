import os
import math
import numpy as np
from cytomine import CytomineJob
from cytomine.models import AttachedFile, ImageInstanceCollection, Job, Property
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from unet import load_data, create_unet


def main(argv):
    # 0. Initialize Cytomine client and job
    with CytomineJob.from_cli(argv) as cj:
        cj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")

        # 1. Create working directories on the machine:
        # - WORKING_PATH/in: input images
        # - WORKING_PATH/out: output images
        # - WORKING_PATH/ground_truth: ground truth images
        # - WORKING_PATH/tmp: temporary path
        base_path = "{}".format(os.getenv("HOME"))
        gt_suffix = "_lbl"
        working_path = os.path.join(base_path, str(cj.job.id))
        in_path = os.path.join(working_path, "in")
        out_path = os.path.join(working_path, "out")
        gt_path = os.path.join(working_path, "ground_truth")
        tmp_path = os.path.join(working_path, "tmp")

        if not os.path.exists(working_path):
            os.makedirs(working_path)
            os.makedirs(in_path)
            os.makedirs(out_path)
            os.makedirs(gt_path)
            os.makedirs(tmp_path)

        # 2. Download the images (first input, then ground truth image)
        cj.job.update(progress=1, statusComment="Downloading images (to {})...".format(in_path))
        image_instances = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
        input_images = [i for i in image_instances if gt_suffix not in i.originalFilename]
        gt_images = [i for i in image_instances if gt_suffix in i.originalFilename]

        for input_image in input_images:
            input_image.download(os.path.join(in_path, "{id}.tif"))

        for gt_image in gt_images:
            related_name = gt_image.originalFilename.replace(gt_suffix, '')
            related_image = [i for i in input_images if related_name == i.originalFilename]
            if len(related_image) == 1:
                gt_image.download(os.path.join(gt_path, "{}.tif".format(related_image[0].id)))

        # 3. Call the image analysis workflow using the run script
        cj.job.update(progress=25, statusComment="Launching workflow...")

        # load data
        cj.job.update(progress=30, statusComment="Workflow: preparing data...")
        dims = cj.parameters.image_height, cj.parameters.image_width, cj.parameters.n_channels

        # load input images
        imgs = load_data(cj, dims, in_path, **{
            "start": 35, "end": 45, "period": 0.1,
            "prefix": "Workflow: load training input images"
        })
        train_mean = np.mean(imgs)
        train_std = np.std(imgs)
        imgs -= train_mean
        imgs /= train_std

        # load masks
        masks = load_data(cj, dims, gt_path, **{
            "start": 45, "end": 55, "period": 0.1,
            "prefix": "Workflow: load training masks images"
        })

        cj.job.update(progress=56, statusComment="Workflow: build model...")
        unet = create_unet(dims)
        unet.compile(optimizer=Adam(lr=cj.parameters.learning_rate), loss='binary_crossentropy')

        cj.job.update(progress=60, statusComment="Workflow: prepare training...")
        datagen = ImageDataGenerator(
            rotation_range=cj.parameters.aug_rotation,
            width_shift_range=cj.parameters.aug_width_shift,
            height_shift_range=cj.parameters.aug_height_shift,
            shear_range=cj.parameters.aug_shear_range,
            horizontal_flip=cj.parameters.aug_hflip,
            vertical_flip=cj.parameters.aug_vflip)

        weight_filepath = os.path.join(tmp_path, 'weights.hdf5')
        callbacks = [
            EarlyStopping(monitor='loss', patience=5, verbose=0),
            ModelCheckpoint(weight_filepath, monitor='loss', save_best_only=True)
        ]

        cj.job.update(progress=65, statusComment="Workflow: train...")
        unet.fit_generator(datagen.flow(imgs, masks, batch_size=cj.parameters.batch_size, seed=42),
                           steps_per_epoch=math.ceil(imgs.shape[0] / cj.parameters.batch_size),
                           epochs=cj.parameters.epochs, callbacks=callbacks)

        # save model and metadata
        cj.job.update(progress=85, statusComment="Save model...")
        AttachedFile(
            cj.job,
            domainIdent=cj.job.id,
            filename=weight_filepath,
            domainClassName="be.cytomine.processing.Job"
        ).upload()

        cj.job.update(progress=90, statusComment="Save metadata...")
        Property(cj.job, key="image_width", value=cj.parameters.image_width).save()
        Property(cj.job, key="image_height", value=cj.parameters.image_height).save()
        Property(cj.job, key="n_channels", value=cj.parameters.n_channels).save()
        Property(cj.job, key="train_mean", value=train_mean).save()
        Property(cj.job, key="image_width", value=train_std).save()

        cj.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")



if __name__ == "__main__":
    import sys

    main(sys.argv[1:])