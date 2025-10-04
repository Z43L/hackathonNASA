import h5py


def print_hdf5_structure(file_path):
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print(f"  [attr] {key}: {val}")
    with h5py.File(file_path, "r") as f:
        print("Estructura del archivo HDF5:")
        f.visititems(print_attrs)

if __name__ == "__main__":
    print_hdf5_structure("nasa_data_functional/3B-MO.MS.MRG.3IMERG.20200101-S000000-E235959.01.V07B.HDF5")
